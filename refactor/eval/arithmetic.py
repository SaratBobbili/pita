import json
import os
import math
import copy
import glob
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from models.classifier import CustomLlamaForSequenceClassification
from models.guidance import CustomValueGuidedLogitProcessor, generate_with_classifier_guidance
from scoring.arithmetic import (
    sample_match_strict, numeric_or_symbolic_correctness, process_sample,
    equivalence_partition, compute_majority_vote_correct
)
from training.dataset import (
    read_jsonl, write_jsonl, tokenize_with_chat_template,
    get_parent_directory, resolve_dict_value, kl_divergence, get_average_reward
)


def run_eval(cfg):
    classifier_ckpt_path = cfg.eval.classifier_ckpt
    with open(os.path.join(get_parent_directory(classifier_ckpt_path), 'args.json'), 'r') as f:
        training_args_dict = json.load(f)

    args_dict = dict(cfg.models) if hasattr(cfg.models, '__iter__') else {}
    ta_models = training_args_dict.get('models', training_args_dict)
    ta_dataset = training_args_dict.get('dataset', training_args_dict)
    ref_model_id = resolve_dict_value(args_dict, ta_models, 'ref_model_id')
    classifier_type = resolve_dict_value(args_dict, ta_models, 'classifier_type')
    classifier_model_id = resolve_dict_value(args_dict, ta_models, 'classifier_model_id')
    inference_mode = resolve_dict_value(args_dict, ta_models, 'inference_mode')
    loss_type = resolve_dict_value(args_dict, ta_models, 'loss_type')
    use_bias = bool(ta_models.get('use_bias', 0))
    eta = resolve_dict_value(args_dict, ta_models, 'eta')
    top_k = resolve_dict_value(args_dict, ta_models, 'top_k')
    dtype = resolve_dict_value(args_dict, ta_models, 'dtype')
    num_atoms = ta_models.get('num_atoms', 11)
    V_min = ta_models.get('V_min', 0.0)
    V_max = ta_models.get('V_max', 1.0)

    dataset_type = ta_dataset.get('name', cfg.dataset.name)
    data_path = ta_dataset.get('data_path', cfg.dataset.data_path)
    train_eval_save_path = ta_dataset.get('train_eval_save_path', cfg.dataset.train_eval_save_path)
    temperature = training_args_dict.get('temperature', 0.8)
    top_p = training_args_dict.get('top_p', 0.9)

    batch_size = cfg.eval.batch_size
    num_samples = cfg.eval.num_samples
    max_new_tokens = cfg.eval.max_new_tokens
    seed = cfg.eval.seed
    output_dir = cfg.trainer.output_dir or classifier_ckpt_path

    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
    vocab_size = len(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    do_sample = temperature != 0
    if not do_sample:
        temperature = 1.0

    if dataset_type in ['gsm8k', 'GSM8K']:
        dataset_type_upper = 'GSM8K'
        answer_key = 'answer'
    else:
        dataset_type_upper = 'MATH'
        answer_key = 'solution'

    match_fn_type = cfg.dataset.get('match_fn', 'symbolic')
    match_fn = numeric_or_symbolic_correctness if match_fn_type == 'symbolic' else sample_match_strict
    extract_last_occurrence = True

    with open(train_eval_save_path, 'r') as f:
        train_eval_problems_d = json.load(f)
    original_examples = read_jsonl(data_path)
    inference_eval_examples = [ex for ex in original_examples if train_eval_problems_d.get(ex['problem'], {}).get('split') == 'eval']

    generate_kwargs = {'temperature': temperature, 'top_p': top_p, 'do_sample': do_sample, 'max_new_tokens': max_new_tokens, 'top_k': 0}
    model_loading_kwargs = {}
    if dtype == 'bfloat16':
        model_loading_kwargs['torch_dtype'] = torch.bfloat16

    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, **model_loading_kwargs, device_map=device)
    classifier_model = CustomLlamaForSequenceClassification.from_pretrained(
        classifier_ckpt_path, **model_loading_kwargs, num_labels=vocab_size,
        classifier_type=classifier_type, loss_type=loss_type, use_bias=use_bias,
        device_map=device, num_atoms=num_atoms, V_min=V_min, V_max=V_max)

    ref_model.eval()
    classifier_model.eval()
    torch.set_grad_enabled(False)

    if eta != 0:
        logit_processor = CustomValueGuidedLogitProcessor(eta=eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
                                                          value_classifier=classifier_model, inference_mode=inference_mode,
                                                          top_k=top_k, use_cache=True)
    else:
        logit_processor = CustomValueGuidedLogitProcessor(eta=eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
                                                          value_classifier=classifier_model, inference_mode='disabled',
                                                          top_k=top_k, use_cache=True)

    individual_dir = os.path.join(output_dir, f'individual_eval_inference_eta_{eta}_top_k_{top_k}_temp_{temperature}')
    os.makedirs(individual_dir, exist_ok=True)

    for i in range(num_samples):
        repeat_index = i
        current_seed = seed + 50 * repeat_index
        set_seed(current_seed)
        data_to_infer = copy.deepcopy(inference_eval_examples)
        num_batches = math.ceil(len(data_to_infer) / batch_size)
        for j in tqdm(range(num_batches)):
            batch_start = j * batch_size
            batch_end = min((j + 1) * batch_size, len(data_to_infer))
            current_prompts = [data_to_infer[k]['prompt'] for k in range(batch_start, batch_end)]
            current_inputs, _ = tokenize_with_chat_template(tokenizer, current_prompts, True, device)

            generate_kwargs['output_scores'] = True
            generate_kwargs['return_dict_in_generate'] = True
            current_outputs = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, generate_kwargs, True, False, eta)
            current_outputs_id = current_outputs['sequences']
            current_outputs_text = tokenizer.batch_decode(current_outputs_id, skip_special_tokens=True)
            aligned_model_scores = torch.stack(current_outputs['scores'], dim=1).float()
            del current_outputs
            torch.cuda.empty_cache()

            kl_batch_size = 2
            token_kl_list = []
            for k in range(0, batch_end - batch_start, kl_batch_size):
                output_attn = (current_outputs_id[k:k+kl_batch_size] != tokenizer.pad_token_id).long()
                concat_ids = torch.cat([current_inputs['input_ids'][k:k+kl_batch_size], current_outputs_id[k:k+kl_batch_size]], dim=1)
                concat_attn = torch.cat([current_inputs['attention_mask'][k:k+kl_batch_size], output_attn], dim=1)
                ref_out = ref_model(input_ids=concat_ids, attention_mask=concat_attn)
                ref_logits = ref_out.logits[:, current_inputs['input_ids'].shape[1]-1:-1].float() / temperature
                cur_kl = kl_divergence(aligned_model_scores[k:k+kl_batch_size].to(ref_logits.device), ref_logits)
                cur_kl = cur_kl * output_attn
                token_kl_list.append(cur_kl)
                del ref_out, ref_logits
                torch.cuda.empty_cache()
            token_kl = torch.cat(token_kl_list, dim=0)
            traj_kl = token_kl.sum(dim=1)
            del aligned_model_scores
            torch.cuda.empty_cache()

            for k in range(batch_end - batch_start):
                idx = batch_start + k
                output_path = os.path.join(individual_dir, f'{idx}_r{repeat_index}.json')
                with open(output_path, 'w') as f:
                    json.dump({
                        'prediction': current_outputs_text[k],
                        'token_kl': token_kl[k].cpu().tolist(),
                        'traj_kl': traj_kl[k].item(),
                    }, f)

    prediction_key = 'predictions'
    for i in range(len(inference_eval_examples)):
        for j in range(num_samples):
            path = os.path.join(individual_dir, f'{i}_r{j}.json')
            with open(path, 'r') as f:
                data = json.load(f)
            if j == 0:
                inference_eval_examples[i][prediction_key] = []
                inference_eval_examples[i]['traj_kl'] = []
            inference_eval_examples[i][prediction_key].append(data['prediction'])
            inference_eval_examples[i]['traj_kl'].append(data['traj_kl'])

    for i in range(len(inference_eval_examples)):
        solution_or_answer = str(inference_eval_examples[i][answer_key])
        if dataset_type_upper == 'MATH':
            answer_processed = process_sample(solution_or_answer, None, extract_last_occurrence)
        else:
            answer_processed = solution_or_answer
        all_samples = inference_eval_examples[i][prediction_key]
        all_processed = [process_sample(s, None, extract_last_occurrence) for s in all_samples]
        predictions_correctness = [match_fn(s, answer_processed) if s is not None else False for s in all_processed]
        inference_eval_examples[i]['predictions_correctness'] = predictions_correctness
        inference_eval_examples[i]['pass_at_k'] = any(predictions_correctness)
        partition = equivalence_partition(all_processed, match_fn)
        inference_eval_examples[i]['majority_vote_correct'] = compute_majority_vote_correct(all_processed, predictions_correctness, partition, strict_tie_breaking=False)

    write_jsonl(inference_eval_examples, os.path.join(output_dir, f'inference_eval_results_eta_{eta}_top_k_{top_k}_temp_{temperature}.jsonl'))

    single_sample_accuracy_list = get_average_reward(inference_eval_examples, 'predictions_correctness', 100)
    single_sample_accuracy_mean = np.mean(single_sample_accuracy_list)
    majority_vote_accuracy_mean = np.mean([ex['majority_vote_correct'] for ex in inference_eval_examples])
    pass_k_accuracy_mean = np.mean([ex['pass_at_k'] for ex in inference_eval_examples])

    with open(os.path.join(output_dir, f'reward_stats_eta_{eta}_top_k_{top_k}_temp_{temperature}.json'), 'w') as f:
        json.dump({
            'single_sample_accuracy_mean': single_sample_accuracy_mean,
            'majority_vote_accuracy_mean': majority_vote_accuracy_mean,
            'pass_k_accuracy_mean': pass_k_accuracy_mean
        }, f)

    print(f'single_sample_accuracy_mean: {single_sample_accuracy_mean}')
    print(f'majority_vote_accuracy_mean: {majority_vote_accuracy_mean}')
    print(f'pass_k_accuracy_mean: {pass_k_accuracy_mean}')
