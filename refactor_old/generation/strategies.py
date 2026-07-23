import torch
import copy
import math
from tqdm import tqdm
from transformers import set_seed, DataCollatorForLanguageModeling
from models.guidance import generate_with_classifier_guidance
from training.dataset import tokenize_with_chat_template, get_output_indices
from scoring.arithmetic import quick_evaluate_single, evaluate_preference


class GuidedGenerate:
    """Math datasets: guided generation + random cut + two unguided continuations + DeBERTa scoring."""

    def __init__(self, ref_model, classifier_model, tokenizer, logit_processor, logit_processor_disabled,
                 reward_model, reward_tokenizer, cfg, match_fn, generate_kwargs):
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.logit_processor = logit_processor
        self.logit_processor_disabled = logit_processor_disabled
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.cfg = cfg
        self.match_fn = match_fn
        self.generate_kwargs = generate_kwargs
        self.dataset_type = 'GSM8K' if 'gsm8k' in cfg.dataset.name else 'MATH'
        self.answer_key = cfg.dataset.answer_key
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def collect(self, batch_data, repeat_index, context_index, seed):
        device = next(self.ref_model.parameters()).device
        set_seed(seed)
        current_prompts = [d['prompt'] for d in batch_data]
        current_inputs, _ = tokenize_with_chat_template(self.tokenizer, current_prompts, True, device)
        current_outputs = generate_with_classifier_guidance(
            self.ref_model, self.tokenizer, self.logit_processor,
            current_inputs, self.generate_kwargs, True, False, self.cfg.models.eta)
        current_outputs_text = self.tokenizer.batch_decode(current_outputs, skip_special_tokens=True)

        for k in range(len(current_outputs_text)):
            solution_or_answer = str(batch_data[k][self.answer_key])
            correctness = quick_evaluate_single(self.dataset_type, solution_or_answer, None, True, self.match_fn, current_outputs_text[k])
            batch_data[k].setdefault('fully_guided_predictions', []).append(current_outputs_text[k])
            batch_data[k].setdefault('fully_guided_predictions_correctness', []).append(correctness)

        outputs_end_indices = get_output_indices(current_outputs, self.tokenizer.eos_token_id)
        outputs_lengths = outputs_end_indices + 1
        random_cut_locations = torch.floor(torch.rand(outputs_lengths.size()).to(outputs_lengths.device) * outputs_lengths).int()
        skip_inference_indices = [k for k in range(len(outputs_lengths)) if random_cut_locations[k] + 1 == outputs_lengths[k]]

        queries = [current_inputs['input_ids'][k].masked_select(current_inputs['attention_mask'][k].to(torch.bool)) for k in range(len(current_inputs['input_ids']))]
        partial_responses = [current_outputs[k][:random_cut_locations[k] + 1] for k in range(len(current_outputs))]
        prompt_partial_response_input_ids = [torch.cat([q, r]) for q, r in zip(queries, partial_responses)]

        non_skip_data = [{'input_ids': prompt_partial_response_input_ids[k], 'attention_mask': torch.ones_like(prompt_partial_response_input_ids[k])} for k in range(len(prompt_partial_response_input_ids)) if k not in skip_inference_indices]
        if non_skip_data:
            prompt_partial_response_input_data = self.data_collator(non_skip_data)
            prompt_partial_response_input_data.pop('labels', None)
            prompt_partial_response_input_data = prompt_partial_response_input_data.to(device)

            first_outputs = generate_with_classifier_guidance(self.ref_model, self.tokenizer, self.logit_processor_disabled, prompt_partial_response_input_data, self.generate_kwargs, True, False, 0)
            first_end_indices = get_output_indices(first_outputs, self.tokenizer.eos_token_id)
            first_lengths = first_end_indices + 1

            set_seed(seed + 1)
            second_outputs = generate_with_classifier_guidance(self.ref_model, self.tokenizer, self.logit_processor_disabled, prompt_partial_response_input_data, self.generate_kwargs, True, False, 0)
            second_end_indices = get_output_indices(second_outputs, self.tokenizer.eos_token_id)
            second_lengths = second_end_indices + 1
        else:
            first_outputs = second_outputs = None
            first_lengths = second_lengths = None

        start_counter = 0
        for k in range(len(queries)):
            if k in skip_inference_indices:
                partial_guided_response_pair = []
            else:
                first_resp = first_outputs[start_counter][:first_lengths[start_counter]].tolist()
                second_resp = second_outputs[start_counter][:second_lengths[start_counter]].tolist()
                partial_guided_response_pair = [first_resp, second_resp]
                start_counter += 1

            if partial_guided_response_pair:
                first_pred_tok = partial_responses[k].tolist() + partial_guided_response_pair[0]
                second_pred_tok = partial_responses[k].tolist() + partial_guided_response_pair[1]
                first_pred = self.tokenizer.decode(first_pred_tok, skip_special_tokens=True)
                second_pred = self.tokenizer.decode(second_pred_tok, skip_special_tokens=True)

                question = self.tokenizer.decode(prompt_partial_response_input_ids[k].tolist())
                inputs_r1 = self.reward_tokenizer(question, first_pred, return_tensors='pt')
                score1 = self.reward_model(**inputs_r1).logits[0].cpu().item()
                inputs_r2 = self.reward_tokenizer(question, second_pred, return_tensors='pt')
                score2 = self.reward_model(**inputs_r2).logits[0].cpu().item()
                soft_pref = [1, 0] if score1 > score2 else [0, 1]

                solution_or_answer = str(batch_data[k][self.answer_key])
                preference = evaluate_preference(self.dataset_type, solution_or_answer, None, True, self.match_fn, first_pred, second_pred, soft_pref)
            else:
                preference = batch_data[k]['fully_guided_predictions_correctness'][-1]
                first_pred = current_outputs_text[k]
                second_pred = current_outputs_text[k]
                partial_guided_response_pair = []

            batch_data[k].setdefault('partial_guided_prompts_tokenized', []).append(prompt_partial_response_input_ids[k].tolist())
            batch_data[k].setdefault('partial_guided_prompts', []).append(self.tokenizer.decode(prompt_partial_response_input_ids[k].tolist()))
            batch_data[k].setdefault('num_response_tokens_in_partial_guided_prompts', []).append(random_cut_locations[k].item() + 1)
            batch_data[k].setdefault('partial_guided_responses_tokenized', []).append(partial_guided_response_pair)
            batch_data[k].setdefault('partial_guided_predictions', []).append([first_pred, second_pred])
            batch_data[k].setdefault('partial_guided_predictions_correctness', []).append(preference)

        return batch_data


class GuidedPairwise:
    """Math pref variant: guided generation + random cut + one unguided continuation + OpenMath2 scoring."""

    def __init__(self, ref_model, classifier_model, tokenizer, logit_processor, logit_processor_disabled,
                 scoring_model, scoring_tokenizer, cfg, match_fn, generate_kwargs):
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.logit_processor = logit_processor
        self.logit_processor_disabled = logit_processor_disabled
        self.scoring_model = scoring_model
        self.scoring_tokenizer = scoring_tokenizer
        self.cfg = cfg
        self.match_fn = match_fn
        self.generate_kwargs = generate_kwargs
        self.dataset_type = 'GSM8K' if 'gsm8k' in cfg.dataset.name else 'MATH'
        self.answer_key = cfg.dataset.answer_key
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def collect(self, batch_data, repeat_index, context_index, seed):
        device = next(self.ref_model.parameters()).device
        set_seed(seed)
        current_prompts = [d['prompt'] for d in batch_data]
        current_inputs, _ = tokenize_with_chat_template(self.tokenizer, current_prompts, True, device)
        current_outputs = generate_with_classifier_guidance(
            self.ref_model, self.tokenizer, self.logit_processor,
            current_inputs, self.generate_kwargs, True, False, self.cfg.models.eta)
        current_outputs_text = self.tokenizer.batch_decode(current_outputs, skip_special_tokens=True)

        for k in range(len(current_outputs_text)):
            solution_or_answer = str(batch_data[k][self.answer_key])
            correctness = quick_evaluate_single(self.dataset_type, solution_or_answer, None, True, self.match_fn, current_outputs_text[k])
            batch_data[k].setdefault('fully_guided_predictions', []).append(current_outputs_text[k])
            batch_data[k].setdefault('fully_guided_predictions_correctness', []).append(correctness)

        outputs_end_indices = get_output_indices(current_outputs, self.tokenizer.eos_token_id)
        outputs_lengths = outputs_end_indices + 1
        random_cut_locations = torch.floor(torch.rand(outputs_lengths.size()).to(outputs_lengths.device) * outputs_lengths).int()
        skip_inference_indices = [k for k in range(len(outputs_lengths)) if random_cut_locations[k] + 1 == outputs_lengths[k]]

        queries = [current_inputs['input_ids'][k].masked_select(current_inputs['attention_mask'][k].to(torch.bool)) for k in range(len(current_inputs['input_ids']))]
        partial_responses = [current_outputs[k][:random_cut_locations[k] + 1] for k in range(len(current_outputs))]
        prompt_partial_response_input_ids = [torch.cat([q, r]) for q, r in zip(queries, partial_responses)]

        non_skip_data = [{'input_ids': prompt_partial_response_input_ids[k], 'attention_mask': torch.ones_like(prompt_partial_response_input_ids[k])} for k in range(len(prompt_partial_response_input_ids)) if k not in skip_inference_indices]
        if non_skip_data:
            prompt_partial_response_input_data = self.data_collator(non_skip_data)
            prompt_partial_response_input_data.pop('labels', None)
            prompt_partial_response_input_data = prompt_partial_response_input_data.to(device)

            set_seed(seed + 10 * context_index)
            unguided_outputs = generate_with_classifier_guidance(self.ref_model, self.tokenizer, self.logit_processor_disabled, prompt_partial_response_input_data, self.generate_kwargs, True, False, 0)
            unguided_end_indices = get_output_indices(unguided_outputs, self.tokenizer.eos_token_id)
            unguided_lengths = unguided_end_indices + 1
        else:
            unguided_outputs = None
            unguided_lengths = None

        start_counter = 0
        for k in range(len(queries)):
            if k in skip_inference_indices:
                partial_guided_responses_tok = []
                partial_pred = current_outputs_text[k]
            else:
                resp_tok = unguided_outputs[start_counter][:unguided_lengths[start_counter]].tolist()
                partial_guided_responses_tok = resp_tok
                partial_pred_tok = partial_responses[k].tolist() + resp_tok
                partial_pred = self.tokenizer.decode(partial_pred_tok, skip_special_tokens=True)
                start_counter += 1

            solution_or_answer = str(batch_data[k][self.answer_key])
            partial_solution = batch_data[k]['prompt'] + " " + partial_pred
            fully_guided_solution = batch_data[k]['prompt'] + " " + current_outputs_text[k]
            partial_inputs = self.scoring_tokenizer(partial_solution, return_tensors="pt")
            full_inputs = self.scoring_tokenizer(fully_guided_solution, return_tensors="pt")
            partial_loss = self.scoring_model(input_ids=partial_inputs["input_ids"], labels=partial_inputs["input_ids"]).loss
            full_loss = self.scoring_model(input_ids=full_inputs["input_ids"], labels=full_inputs["input_ids"]).loss
            soft_pref = (torch.exp(partial_loss) < torch.exp(full_loss)).item()

            preference = evaluate_preference(self.dataset_type, solution_or_answer, None, True, self.match_fn, partial_pred, current_outputs_text[k], soft_pref)

            batch_data[k].setdefault('partial_guided_prompts_tokenized', []).append(prompt_partial_response_input_ids[k].tolist())
            batch_data[k].setdefault('partial_guided_prompts', []).append(self.tokenizer.decode(prompt_partial_response_input_ids[k].tolist()))
            batch_data[k].setdefault('num_response_tokens_in_partial_guided_prompts', []).append(random_cut_locations[k].item() + 1)
            batch_data[k].setdefault('partial_guided_responses_tokenized', []).append(partial_guided_responses_tok)
            batch_data[k].setdefault('partial_guided_predictions', []).append(partial_pred)
            batch_data[k].setdefault('partial_guided_predictions_correctness', []).append(preference)

        return batch_data


class OfflinePairs:
    """Preference datasets (HH-RLHF, AlpacaEval): no generation, just tokenize existing pairs."""

    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def collect(self, batch_data, repeat_index, context_index, seed):
        device = 'cpu'
        for k in range(len(batch_data)):
            prompt = batch_data[k]['prompt']
            output_1 = batch_data[k]['output_1']
            output_2 = batch_data[k]['output_2']
            preference = batch_data[k]['preference']

            inputs, _ = tokenize_with_chat_template(self.tokenizer, [prompt], True, device)
            output_1_tok = self.tokenizer(output_1, add_special_tokens=False)["input_ids"]
            output_2_tok = self.tokenizer(output_2, add_special_tokens=False)["input_ids"]

            batch_data[k].setdefault('fully_guided_predictions', []).append(output_1)
            batch_data[k].setdefault('fully_guided_predictions_correctness', []).append(preference == 1)
            batch_data[k].setdefault('fully_guided_responses_tokenized', []).append(output_1_tok)
            batch_data[k].setdefault('partial_guided_prompts_tokenized', []).append(inputs['input_ids'][0].tolist())
            batch_data[k].setdefault('partial_guided_prompts', []).append(self.tokenizer.decode(inputs['input_ids'][0].tolist()))
            batch_data[k].setdefault('partial_guided_responses_tokenized', []).append(output_2_tok)
            batch_data[k].setdefault('partial_guided_predictions', []).append(output_2)
            batch_data[k].setdefault('partial_guided_predictions_correctness', []).append(preference == 2)

        return batch_data
