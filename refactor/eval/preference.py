import json
import os
import math
import copy
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed

from models.classifier import CustomLlamaForSequenceClassification
from models.guidance import CustomValueGuidedLogitProcessor, generate_with_classifier_guidance
from training.dataset import (
    read_jsonl, write_jsonl, write_json_array, tokenize_with_chat_template,
    get_parent_directory, resolve_dict_value, perplexity_with_classifier_guidance
)


def run_eval(cfg):
    classifier_ckpt_path = cfg.eval.classifier_ckpt
    with open(os.path.join(get_parent_directory(classifier_ckpt_path), 'args.json'), 'r') as f:
        training_args_dict = json.load(f)

    models_dict = dict(cfg.models) if hasattr(cfg.models, '__iter__') else {}
    ta_models = training_args_dict.get('models', training_args_dict)
    ref_model_id = resolve_dict_value(models_dict, ta_models, 'ref_model_id')
    classifier_type = resolve_dict_value(models_dict, ta_models, 'classifier_type')
    classifier_model_id = resolve_dict_value(models_dict, ta_models, 'classifier_model_id')
    inference_mode = resolve_dict_value(models_dict, ta_models, 'inference_mode')
    loss_type = resolve_dict_value(models_dict, ta_models, 'loss_type')
    use_bias = bool(ta_models.get('use_bias', 0))
    eta = resolve_dict_value(models_dict, ta_models, 'eta')
    top_k = resolve_dict_value(models_dict, ta_models, 'top_k')
    dtype = resolve_dict_value(models_dict, ta_models, 'dtype')
    num_atoms = training_args_dict.get('num_atoms', 11)
    V_min = training_args_dict.get('V_min', 0.0)
    V_max = training_args_dict.get('V_max', 1.0)
    temperature = training_args_dict.get('temperature', 0.8)
    top_p = training_args_dict.get('top_p', 0.9)

    batch_size = cfg.eval.batch_size
    num_samples = cfg.eval.num_samples
    seed = cfg.eval.seed
    output_dir = cfg.trainer.output_dir or classifier_ckpt_path

    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
    vocab_size = AutoConfig.from_pretrained(classifier_ckpt_path).vocab_size
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    do_sample = temperature != 0
    if not do_sample:
        temperature = 1.0

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

    eval_type = cfg.dataset.eval_type
    if eval_type == "preference_ppl":
        _run_preference_ppl(cfg, ref_model, tokenizer, logit_processor, device, eta, top_k, temperature,
                           batch_size, num_samples, seed, output_dir)
    elif eval_type == "preference_gen":
        _run_preference_gen(cfg, ref_model, tokenizer, logit_processor, device, eta, top_k, temperature,
                           top_p, do_sample, batch_size, num_samples, seed, output_dir)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")


def _run_preference_ppl(cfg, ref_model, tokenizer, logit_processor, device, eta, top_k, temperature,
                        batch_size, num_samples, seed, output_dir):
    data_path = cfg.dataset.data_path
    if os.path.isfile(data_path):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict):
            inference_eval_examples = []
            for prompt_text, value in raw_data.items():
                ex = copy.deepcopy(value)
                ex['instruction'] = prompt_text
                inference_eval_examples.append(ex)
        else:
            inference_eval_examples = raw_data
    else:
        import datasets as hf_datasets
        inference_eval_examples = hf_datasets.load_dataset(data_path, split="eval").to_list()

    inference_eval_examples = [ex for ex in inference_eval_examples if ex.get('split', 'eval') == 'eval']
    individual_dir = os.path.join(output_dir, f'individual_eval_inference_{seed}_eta_{eta}_top_k_{top_k}_temp_{temperature}')
    os.makedirs(individual_dir, exist_ok=True)

    prompt_key = 'instruction' if 'instruction' in inference_eval_examples[0] else list(inference_eval_examples[0].keys())[0]

    for i in range(num_samples):
        repeat_index = i
        set_seed(seed + 50 * repeat_index)
        num_batches = math.ceil(len(inference_eval_examples) / batch_size)
        for j in tqdm(range(num_batches)):
            batch_start = j * batch_size
            batch_end = min((j + 1) * batch_size, len(inference_eval_examples))
            current_prompts = [inference_eval_examples[k][prompt_key] for k in range(batch_start, batch_end)]
            current_inputs, _ = tokenize_with_chat_template(tokenizer, current_prompts, True, device)

            response_1_texts = [str(inference_eval_examples[k].get('output_1', '')) for k in range(batch_start, batch_end)]
            response_2_texts = [str(inference_eval_examples[k].get('output_2', '')) for k in range(batch_start, batch_end)]
            r1_inputs = tokenizer(response_1_texts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
            r2_inputs = tokenizer(response_2_texts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)

            ppl_1 = perplexity_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, r1_inputs, eta)
            ppl_2 = perplexity_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, r2_inputs, eta)

            predicted_pref = torch.where(ppl_1 <= ppl_2, 1, 2)
            true_pref = torch.tensor([int(inference_eval_examples[k].get('preference', 1)) for k in range(batch_start, batch_end)], device=device)
            is_success = predicted_pref.eq(true_pref)

            for k in range(batch_end - batch_start):
                idx = batch_start + k
                path = os.path.join(individual_dir, f'{idx}_r{repeat_index}.json')
                with open(path, 'w') as f:
                    json.dump({
                        'ppl_output_1': float(ppl_1[k].item()),
                        'ppl_output_2': float(ppl_2[k].item()),
                        'predicted_preference': int(predicted_pref[k].item()),
                        'true_preference': int(true_pref[k].item()),
                        'is_success': int(is_success[k].item()),
                    }, f)

    all_successes = []
    for i in range(len(inference_eval_examples)):
        for j in range(num_samples):
            path = os.path.join(individual_dir, f'{i}_r{j}.json')
            with open(path, 'r') as f:
                data = json.load(f)
            all_successes.append(data['is_success'])

    win_rate = float(np.mean(all_successes))
    write_jsonl(inference_eval_examples, os.path.join(output_dir, f'inference_eval_results_{seed}_eta_{eta}_top_k_{top_k}_temp_{temperature}.jsonl'))
    with open(os.path.join(output_dir, f'reward_stats_{seed}_eta_{eta}_top_k_{top_k}_temp_{temperature}.json'), 'w') as f:
        json.dump({'win_rate': win_rate, 'num_examples': len(inference_eval_examples), 'num_samples': num_samples}, f, indent=2)
    print(f'Win rate: {win_rate}')


def _run_preference_gen(cfg, ref_model, tokenizer, logit_processor, device, eta, top_k, temperature,
                        top_p, do_sample, batch_size, num_samples, seed, output_dir):
    import datasets as hf_datasets
    eval_data = hf_datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval").to_list()
    max_new_tokens = cfg.eval.max_new_tokens
    generate_kwargs = {'temperature': temperature, 'top_p': top_p, 'do_sample': do_sample, 'max_new_tokens': max_new_tokens, 'top_k': 0}

    individual_dir = os.path.join(output_dir, f'individual_eval_inference_{seed}_eta_{eta}_top_k_{top_k}_temp_{temperature}')
    os.makedirs(individual_dir, exist_ok=True)

    for i in range(num_samples):
        repeat_index = i
        set_seed(seed + 50 * repeat_index)
        num_batches = math.ceil(len(eval_data) / batch_size)
        for j in tqdm(range(num_batches)):
            batch_start = j * batch_size
            batch_end = min((j + 1) * batch_size, len(eval_data))
            current_prompts = [eval_data[k]['instruction'] for k in range(batch_start, batch_end)]
            current_inputs, _ = tokenize_with_chat_template(tokenizer, current_prompts, True, device)
            current_outputs = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, generate_kwargs, True, True, eta)

            for k in range(batch_end - batch_start):
                idx = batch_start + k
                path = os.path.join(individual_dir, f'{idx}_r{repeat_index}.json')
                with open(path, 'w') as f:
                    json.dump({'output': current_outputs[k]}, f)

    model_outputs = []
    for i in range(len(eval_data)):
        outputs_for_example = []
        for j in range(num_samples):
            path = os.path.join(individual_dir, f'{i}_r{j}.json')
            with open(path, 'r') as f:
                data = json.load(f)
            outputs_for_example.append(data['output'])
        model_outputs.append({
            'instruction': eval_data[i]['instruction'],
            'output': outputs_for_example[0],
            'generator': f'pita_eta_{eta}',
        })

    write_json_array(model_outputs, os.path.join(output_dir, 'model_outputs.json'))
    print(f'Saved {len(model_outputs)} model outputs to {output_dir}/model_outputs.json')
