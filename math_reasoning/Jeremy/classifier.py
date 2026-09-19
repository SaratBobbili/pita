from transformers import LlamaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LogitsProcessor
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position


class CustomLlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config, loss_type, use_bias, classifier_type, *, num_atoms: int = 11, V_min: float = 0.0, V_max: float = 1.0):
        assert classifier_type in ["Q", "V"]
        print("Creating classifier of type ", classifier_type)
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier_type = classifier_type
        self.model = LlamaModel(config)
        # num_labels should be vocab_size
        if loss_type == "mse":
            self.loss_fct = MSELoss(reduction="none")
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, 1, bias=use_bias)
        elif loss_type == "bce":
            self.loss_fct = BCEWithLogitsLoss(reduction="none")
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, 1, bias=use_bias)
        elif loss_type == "mle":
            self.num_atoms = num_atoms
            self.V_min = V_min
            self.V_max = V_max
            # linspace includes V_min (i=0) and V_max (i=-1)
            self.atoms = torch.linspace(self.V_min, self.V_max, self.num_atoms).float()
            if self.classifier_type == "Q":
                self.score = nn.Linear(config.hidden_size, self.num_labels * self.num_atoms, bias=use_bias)
            elif self.classifier_type == "V":
                self.score = nn.Linear(config.hidden_size, self.num_atoms, bias=use_bias)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}.")

        self.loss_type = loss_type
        self.use_bias = use_bias
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def zero_init_classifier(self):
        nn.init.zeros_(self.score.weight)
        if self.use_bias:
            nn.init.zeros_(self.score.bias)

    def calculate_loss(self, logits, labels, loss_weights, loss_mask):
        # logits: [bs, seqlen, 1] or [bs, seqlen, num_atoms]
        # loss_mask has same shape as logits
        assert len(logits.shape) == 3
        bs, seqlen, _ = logits.shape
        assert loss_mask.shape == (bs, seqlen)
        assert labels.shape == (bs,)
        assert loss_weights.shape == (bs,)

        if self.loss_type == "mse":
            # for MSE we will explicitly regress with sigmoid
            relevant_logits = torch.sigmoid(logits).squeeze(-1)  # [bs, seqlen]
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = self.loss_fct(relevant_logits, labels_expanded.to(relevant_logits.dtype))
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)  # [bs]
        elif self.loss_type == "bce":
            # for BCE, sigmoid is implicitly applied in the loss function
            assert logits.shape[2] == 1
            logits = logits.squeeze(-1)
            labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)
            loss = self.loss_fct(logits, labels_expanded)
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)  # [bs]
        elif self.loss_type == "mle":
            log_pmfs = F.log_softmax(logits, dim=-1)  # [bs, seqlen, num_atoms]
            # round labels to nearest atom
            # and clamp underflow and overflows to V_min, V_max
            label_indices = torch.round(labels * (self.num_atoms - 1)).long()
            label_indices = torch.clamp(label_indices, 0, self.num_atoms - 1)  # [bs]
            # for each batch_idx, select the corresponding index of num_atoms
            loss = -log_pmfs[torch.arange(bs), :, label_indices]  # [bs, seqlen]
            loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)  # [bs]
        else:
            raise RuntimeError("Impossible to reach.")
        assert loss.shape == loss_weights.shape
        loss = loss * loss_weights
        loss = loss.mean()
        return loss

    def calculate_predictions(self, logits):
        # use logits to predict E[R]
        # logits: [bs, seqlen, 1] or [bs, seqlen, num_atoms]
        bs, seqlen, num_labels = logits.shape
        if self.loss_type in ["mse", "bce"]:
            # for both MSE and BCE, we will use sigmoid to predict the value
            return torch.sigmoid(logits).squeeze(-1)  # [bs, seqlen]
        elif self.loss_type == "mle":
            pmfs = torch.softmax(logits, dim=-1)  # [bs, seqlen, num_atoms]
            if self.atoms.device != pmfs.device:
                self.atoms = self.atoms.to(pmfs.device)
            return (pmfs * self.atoms).sum(dim=-1)  # [bs, seqlen]
        else:
            raise NotImplementedError()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            loss_weights: Optional[torch.FloatTensor] = None,
            logit_indices: Optional[torch.LongTensor] = None,
            loss_mask: Optional[torch.BoolTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bs = input_ids.size(0)

        """
        Two uses cases.
        If training:
        - input_ids: [bs, seq_len] where seq_len is both input and target ids
        - logit_indices is None in this case
        - loss_mask [bs, seq_len] which gives indices for the target ids

        If inference:
        - input_ids: [bs, seq_len] for pre-fill, [bs, 1] for AR

        """
        if self.classifier_type == "Q":
            if labels is not None:
                # training
                bs, seqlen = input_ids.shape
                transformer_outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # force assume we are using left padding and sequence_lengths is -1
                hidden_states = transformer_outputs[0]  # [bs, seqlen, hidden_dim]
                # index first and then convert to float -- keep as bf16 for now
                logits = self.score(hidden_states)  # [bs, seqlen, vocab_size] or [bs, seqlen, vocab_size * num_atoms]
                if self.loss_type in ["mse", "bce"]:
                    logits = logits.unsqueeze(-1)  # [bs, seqlen, vocab_size, 1]
                elif self.loss_type == "mle":
                    logits = logits.view(bs, seqlen, self.num_labels, self.num_atoms)  # [bs, seqlen, vocab_size, num_atoms]
                indexed_logits = logits[:, :-1][torch.arange(bs)[:, None], torch.arange(seqlen-1), input_ids[:, 1:]]
                indexed_logits = indexed_logits.float()
                loss = self.calculate_loss(indexed_logits, labels, loss_weights, loss_mask[:, 1:])
                return SequenceClassifierOutputWithPast(loss=loss, logits=indexed_logits)
            else:
                # inference
                # in first pass, input_ids: [bs, seqlen]
                # in subsequent passes, input_ids: [bs, 1]
                bs, _ = input_ids.shape
                top_k = self.num_labels
                if logit_indices is not None:
                    top_k = logit_indices.size(1)
                transformer_outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # force assume we are using left padding and sequence_lengths is -1
                hidden_states = transformer_outputs[0]  # [bs, seqlen, hidden_dim] or [bs, 1, hidden_dim]
                hidden_states = hidden_states[:, -1]  # [bs, hidden_size]
                logits = self.score(hidden_states)  # [bs, vocab_size * (1 or num_atoms or 2)]

                if self.loss_type in ["mse", "bce"]:
                    # logits: [bs, vocab_size]
                    if logit_indices is not None:
                        # index with logit_indices
                        # [bs, vocab_size] -> [bs, top_k]
                        logits = logits[torch.arange(bs)[:, None], logit_indices]
                elif self.loss_type == "mle":
                    # logits: [bs, vocab_size * num_atoms]
                    if logit_indices is not None:
                        # index with logit_indices
                        # [bs, vocab_size * num_atoms] -> [bs, top_k, num_atoms]
                        offsets = torch.arange(self.num_atoms, device=logit_indices.device)
                        expanded = logit_indices.unsqueeze(-1) * self.num_atoms + offsets  # [bs, top_k, num_atoms]
                        expanded = expanded.view(bs, -1)  # [bs, top_k * num_atoms]
                        logits = logits[torch.arange(bs)[:, None], expanded]  # [bs, top_k * num_atoms]
                    logits = logits.float().view(bs, top_k, self.num_atoms)  # [bs, top_k, num_atoms]

                return SequenceClassifierOutputWithPast(
                    logits=logits,
                    past_key_values=transformer_outputs.past_key_values,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )

        elif self.classifier_type == "V":
            assert return_dict, "V must return dict"
            if labels is not None:
                # training
                assert logit_indices is None
                assert loss_mask is not None
                transformer_outputs = self.model(input_ids, attention_mask=attention_mask)
                # force assume we are using left padding and sequence_lengths is -1
                hidden_states = transformer_outputs[0]  # [bs, seq_len, hidden_size]
                logits = self.score(hidden_states).float()  # [bs, seq_len, 1] or [bs, seq_len, num_atoms]
                loss = self.calculate_loss(logits, labels, loss_weights, loss_mask)
                return SequenceClassifierOutputWithPast(loss=loss, logits=logits)
            else:
                # inference
                top_k = logit_indices.size(1)
                transformer_outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
                # use the same past_key_values for all top_k computation
                output_past_key_values = transformer_outputs.past_key_values
                dtype, device = output_past_key_values[0][0].dtype, output_past_key_values[0][0].device
                min_dtype = torch.finfo(dtype).min
                next_input_ids = logit_indices.to(input_ids.device)
                expanded_attention_mask = torch.cat([attention_mask, torch.ones((bs, top_k), dtype=torch.long, device=attention_mask.device)], dim=1)
                cache_position = torch.arange(attention_mask.shape[1], expanded_attention_mask.shape[1], device=device)
                actual_position_ids = (torch.ones((1, top_k)) * attention_mask.shape[1]).to(dtype=attention_mask.dtype, device=device)
                actual_attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(expanded_attention_mask, top_k, expanded_attention_mask.shape[1], dtype=dtype, device=device, min_dtype=min_dtype, cache_position=cache_position, batch_size=input_ids.shape[0])
                diagonal_mask = torch.full((top_k, top_k), min_dtype)
                diagonal_mask.fill_diagonal_(0)
                diagonal_mask = diagonal_mask.to(dtype=actual_attention_mask.dtype, device=device)
                actual_attention_mask[:, :, :, -top_k:] = diagonal_mask
                transformer_outputs = self.model(next_input_ids, attention_mask=actual_attention_mask, position_ids=actual_position_ids,
                                                 past_key_values=output_past_key_values, use_cache=True, cache_position=cache_position)
                hidden_states = transformer_outputs[0]
                hidden_states = hidden_states[:, -top_k:]
                logits = self.score(hidden_states)
                if self.loss_type == "mle":
                    assert logits.shape == (bs, top_k, self.num_atoms)
                else:
                    logits = logits.squeeze(-1)
                return SequenceClassifierOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=output_past_key_values,
                )


def log1p_exp(x):
    return torch.logaddexp(x, torch.tensor(0.0).to(x.device))


class CustomValueGuidedLogitProcessor(LogitsProcessor):

    def __init__(self, eta, ref_model, ref_model_tokenizer, value_classifier, inference_mode, top_k, cd_baseline=0, use_cache=True):
        self.eta = eta  # reciprocal of KL divergence weight, larger it is, smaller KL is. ignored when inference_mode is 'expectation'
        self.ref_model = ref_model
        self.ref_model_tokenizer = ref_model_tokenizer  # tokenizer for ref_model and value_classifier
        self.inference_mode = inference_mode  # inference mode for classifier
        self.modify_top_k = top_k  # only modify the top k logits
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
            # Update attention_mask and input_ids
            attention_mask = torch.cat(
                [self.classifier_state["attention_mask"], torch.ones_like(input_ids[:, -1:], dtype=torch.long)], dim=1)
            if not self.classifier_state["use_cache"]:
                input_ids = torch.cat([self.classifier_state["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]  # due to cache, we only need the last token
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
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()  # [bs, top_k, num_atoms]
            log_pmfs = F.log_softmax(classifier_logits, dim=-1)  # [bs, top_k, num_atoms]
            atoms = self.value_classifier.atoms.float()  # [num_atoms]
            if atoms.device != log_pmfs.device:
                atoms = atoms.to(log_pmfs.device)

            logit_offset = torch.logsumexp(log_pmfs + self.eta * atoms, dim=-1)  # [bs, top_k]
            logit_offset = logit_offset - logit_offset.min(dim=-1, keepdim=True).values  # [bs, top_k]
            combined_logits = self.modify_top_k_logits(ref_model_logits, logit_offset, top_k_indices)  # [bs, vocab_size]

        elif self.inference_mode == 'expectation':
            classifier_logits = self.get_classifier_values(input_ids, top_k_indices).float()
            if self.cd_baseline:
                logit_offset = self.eta * torch.sigmoid(classifier_logits)
            else:
                #logit_offset = F.logsigmoid(classifier_logits)
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
