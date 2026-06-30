import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
from transformers.generation.logits_process import LogitsProcessorList


def log1p_exp(x):
    return torch.logaddexp(x, torch.tensor(0.0).to(x.device))


class CustomValueGuidedLogitProcessor(LogitsProcessor):

    def __init__(self, eta, ref_model, ref_model_tokenizer, value_classifier, inference_mode, top_k, cd_baseline=0, use_cache=True):
        self.eta = eta
        self.ref_model = ref_model
        self.ref_model_tokenizer = ref_model_tokenizer
        self.inference_mode = inference_mode
        self.modify_top_k = top_k
        assert self.inference_mode in ['expectation', 'bernoulli', 'disabled']
        self.cd_baseline = cd_baseline
        self.value_classifier = value_classifier
        self.loss_type = value_classifier.loss_type
        self.use_cache = use_cache
        self.classifier_state = {"input_ids": None, "attention_mask": None, "use_cache": use_cache,
                                 "past_key_values": None, "first_pass": True}

    def reset_classifier_state(self):
        self.classifier_state = {"input_ids": None, "attention_mask": None, "use_cache": self.use_cache,
                                 "past_key_values": None, "first_pass": True}

    def get_classifier_values(self, input_ids, top_k_indices):
        if self.classifier_state['first_pass']:
            assert self.classifier_state['input_ids'] is None
            assert self.classifier_state['attention_mask'] is None
            assert self.classifier_state['past_key_values'] is None
            self.classifier_state['first_pass'] = False
            self.classifier_state['input_ids'] = input_ids
            pad_token_id = self.ref_model_tokenizer.pad_token_id
            attention_mask = input_ids.ne(pad_token_id).long()
            self.classifier_state['attention_mask'] = attention_mask.to(input_ids.dtype)
        else:
            attention_mask = torch.cat(
                [self.classifier_state["attention_mask"], torch.ones_like(input_ids[:, -1:], dtype=torch.long)], dim=1)
            if not self.classifier_state["use_cache"]:
                input_ids = torch.cat([self.classifier_state["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.classifier_state["input_ids"] = input_ids
            self.classifier_state["attention_mask"] = attention_mask
        with torch.no_grad():
            classifier_outputs = self.value_classifier(input_ids=input_ids, attention_mask=attention_mask,
                                                       use_cache=self.classifier_state["use_cache"], logit_indices=top_k_indices,
                                                       past_key_values=self.classifier_state["past_key_values"])
        if self.classifier_state['use_cache']:
            assert classifier_outputs.past_key_values is not None
            self.classifier_state['past_key_values'] = classifier_outputs.past_key_values
        return classifier_outputs.logits

    def modify_top_k_logits(self, ref_model_logits, logit_offset, top_k_indices):
        return torch.scatter_add(ref_model_logits, 1, top_k_indices.to(ref_model_logits.device), logit_offset)

    def __call__(self, input_ids, ref_model_logits):
        if self.inference_mode == 'disabled':
            return ref_model_logits

        if self.modify_top_k == -1:
            top_k_indices = torch.arange(ref_model_logits.size(-1)).unsqueeze(0).expand(ref_model_logits.size(0), -1)
        else:
            _, top_k_indices = torch.topk(ref_model_logits, self.modify_top_k, dim=-1)
        if self.loss_type == "mle":
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()
            log_pmfs = F.log_softmax(classifier_logits, dim=-1)
            atoms = self.value_classifier.atoms.float()
            if atoms.device != log_pmfs.device:
                atoms = atoms.to(log_pmfs.device)
            logit_offset = torch.logsumexp(log_pmfs + self.eta * atoms, dim=-1)
            logit_offset = logit_offset - logit_offset.min(dim=-1, keepdim=True).values
            combined_logits = self.modify_top_k_logits(ref_model_logits, logit_offset, top_k_indices)
        elif self.inference_mode == 'expectation':
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()
            if self.cd_baseline:
                logit_offset = self.eta * torch.sigmoid(classifier_logits)
            else:
                ratio = torch.sigmoid(classifier_logits) / (1 - torch.sigmoid(classifier_logits))
                ratio = torch.clamp(ratio, min=1e-6, max=1 - 1e-6)
                logit_offset = self.eta * torch.log(ratio)
            combined_logits = self.modify_top_k_logits(ref_model_logits, logit_offset, top_k_indices)
        elif self.inference_mode == 'bernoulli':
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()
            if self.cd_baseline:
                logit_offset = self.eta * torch.sigmoid(classifier_logits)
            else:
                log_numerator = log1p_exp(self.eta + classifier_logits)
                log_denominator = log1p_exp(classifier_logits)
                logit_offset = log_numerator - log_denominator
            combined_logits = self.modify_top_k_logits(ref_model_logits, logit_offset, top_k_indices)
        else:
            raise ValueError("Invalid inference mode")
        return combined_logits


def generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, inputs, generate_kwargs, return_output_only, return_text, eta):
    logit_processor.reset_classifier_state()
    logit_processors = LogitsProcessorList([logit_processor])
    if eta != 0:
        with torch.no_grad():
            outputs = ref_model.generate(**inputs, logits_processor=logit_processors, pad_token_id=tokenizer.pad_token_id, **generate_kwargs)
    else:
        with torch.no_grad():
            outputs = ref_model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **generate_kwargs)
    if return_output_only:
        if isinstance(outputs, dict) and 'sequences' in outputs:
            outputs['sequences'] = outputs['sequences'][:, inputs['input_ids'].shape[1]:]
        else:
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
    if return_text:
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_outputs
    else:
        return outputs
