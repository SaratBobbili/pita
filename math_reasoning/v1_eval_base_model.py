import argparse
import json
import os
from tqdm import tqdm
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import copy
from accuracy_utils import sample_match_strict, process_sample, numeric_or_symbolic_correctness, \
    equivalence_partition, compute_majority_vote_correct
from utils import read_jsonl, tokenize_with_chat_template, write_jsonl, get_average_reward

parser = argparse.ArgumentParser(description='Evaluate base model without classifier guidance')
parser.add_argument('--ref_model_id', required=True, type=str,
                    help='reference model id e.g., meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--dataset_type', required=True, type=str, help='dataset type: gsm8k or math')
parser.add_argument('--data_path', required=True, type=str, help='path to eval data e.g., dataset/gsm8k_test.jsonl')
parser.add_argument('--train_eval_save_path', required=True, type=str,
                    help='train eval split e.g., dataset/gsm8k_test_eval.json')
parser.add_argument('--batch_size', default=8, type=int, help='batch size for generation')
parser.add_argument('--num_samples', default=8, type=int, help='number of samples per problem')
parser.add_argument('--use_chat_template', default=1, type=int, help='whether to use chat template')
parser.add_argument('--temperature', default=0.8, type=float, help='sampling temperature')
parser.add_argument('--top_p', default=0.9, type=float, help='top p for nucleus sampling')
parser.add_argument('--max_prompt_length', default=-1, type=int, help='max tokens for prompt, -1 for no limit')
parser.add_argument('--max_new_tokens', default=1024, type=int, help='max new tokens to generate')
parser.add_argument('--dtype', default='bfloat16', type=str, help='model dtype: bfloat16 or float32')
parser.add_argument('--match_fn_type', default='symbolic', type=str,
                    help='answer matching function: symbolic or strict')
parser.add_argument('--output_dir', required=True, type=str, help='directory to save results')
parser.add_argument('--force', default=0, type=int, help='force overwrite existing files')
parser.add_argument('--seed', default=47, type=int, help='random seed')

args = parser.parse_args()
print(vars(args))

os.makedirs(args.output_dir, exist_ok=True)
individual_eval_inference_output_dir = os.path.join(args.output_dir, 'individual_eval_inference_base_model')
os.makedirs(individual_eval_inference_output_dir, exist_ok=True)

if args.force:
    os.system('rm -rf {0}/*'.format(individual_eval_inference_output_dir))

# Check if results already exist
results_path = os.path.join(args.output_dir, 'inference_eval_results_base_model.jsonl')
stats_path = os.path.join(args.output_dir, 'reward_stats_base_model.json')
if not args.force and os.path.exists(results_path) and os.path.exists(stats_path):
    print('Output exists, skipping')
    exit(0)

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.ref_model_id)
if tokenizer.pad_token is None:
    assert 'Llama-3' in args.ref_model_id
    tokenizer.pad_token = tokenizer.added_tokens_decoder[128002].content
tokenizer.padding_side = "left"

# Configure sampling
if args.temperature == 0:
    do_sample = False
    temperature = 1.0
else:
    do_sample = True
    temperature = args.temperature

# Load model
model_loading_kwargs = {}
if args.dtype == 'bfloat16':
    model_loading_kwargs['torch_dtype'] = torch.bfloat16
ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_id, **model_loading_kwargs, device_map=device)
ref_model.eval()
torch.set_grad_enabled(False)

# Configure dataset
prediction_key = 'predictions'
if args.dataset_type == 'gsm8k':
    dataset_type = 'GSM8K'
    answer_key = 'answer'
    extract_last_occurrence = True
elif args.dataset_type == 'math':
    dataset_type = 'MATH'
    answer_key = 'solution'
    extract_last_occurrence = True
else:
    raise ValueError('Unknown dataset type: %s' % args.dataset_type)

# Configure matching function
if args.match_fn_type == 'strict':
    match_fn = sample_match_strict
elif args.match_fn_type == 'symbolic':
    match_fn = numeric_or_symbolic_correctness
else:
    raise ValueError('Unknown match function type: %s' % args.match_fn_type)

# Load evaluation data
with open(args.train_eval_save_path, 'r') as f:
    train_eval_problems_d = json.load(f)
original_id_to_eval_id_d = {}
original_examples = read_jsonl(args.data_path)
inference_eval_examples = []
for i in range(len(original_examples)):
    if train_eval_problems_d[original_examples[i]['problem']]['split'] == 'eval':
        inference_eval_examples.append(original_examples[i])
        original_id = train_eval_problems_d[original_examples[i]['problem']]['id']
        original_id_to_eval_id_d[original_id] = len(original_id_to_eval_id_d)

# Skip problems with prompts that are too long
skip_problems = []
for j in range(len(inference_eval_examples)):
    num_toks = len(tokenizer(inference_eval_examples[j]['prompt'])['input_ids'])
    if args.max_prompt_length != -1 and num_toks > args.max_prompt_length:
        skip_problems.append(j)

# Generation config
generate_kwargs = {
    'temperature': temperature,
    'top_p': args.top_p,
    'do_sample': do_sample,
    'max_new_tokens': args.max_new_tokens,
    'top_k': 0
}

# Generate samples
for i in range(args.num_samples):
    repeat_index = i
    current_seed = args.seed + 50 * repeat_index
    set_seed(current_seed)
    print('Repeat {0}'.format(repeat_index))
    
    # Check which problems already have results
    if not args.force:
        existing_indices = []
        for j in range(len(inference_eval_examples)):
            output_path = os.path.join(individual_eval_inference_output_dir, f'{j}_r{repeat_index}.json')
            if os.path.exists(output_path):
                existing_indices.append(j)
    else:
        existing_indices = []
    
    # Collect problems to infer
    data_to_infer = []
    for j in range(len(inference_eval_examples)):
        if j in existing_indices or j in skip_problems:
            continue
        data_to_infer.append(copy.deepcopy(inference_eval_examples[j]))
    
    print('Total problems to infer for repeat {0}: {1}'.format(repeat_index, len(data_to_infer)))
    
    # Process in batches
    num_batches = math.ceil(len(data_to_infer) / args.batch_size)
    for j in tqdm(range(num_batches)):
        batch_start = j * args.batch_size
        batch_end = min((j + 1) * args.batch_size, len(data_to_infer))
        
        # Prepare batch
        current_prompts = [data_to_infer[k]['prompt'] for k in range(batch_start, batch_end)]
        current_inputs, _ = tokenize_with_chat_template(tokenizer, current_prompts,
                                                        args.use_chat_template, device)
        
        # Generate
        current_outputs = ref_model.generate(**current_inputs, **generate_kwargs)
        current_outputs_text = tokenizer.batch_decode(current_outputs, skip_special_tokens=True)
        
        # Save results
        for k in range(batch_end - batch_start):
            original_problem_id = train_eval_problems_d[data_to_infer[batch_start + k]['problem']]['id']
            eval_problem_id = original_id_to_eval_id_d[original_problem_id]
            output_path = os.path.join(individual_eval_inference_output_dir, f'{eval_problem_id}_r{repeat_index}.json')
            assert not os.path.exists(output_path), f"Expect {output_path} to not exist"
            with open(output_path, 'w') as f:
                json.dump({
                    'prediction': current_outputs_text[k],
                }, f)

print('Done inference, now combining results')

# Combine results
for i in range(len(inference_eval_examples)):
    if i in skip_problems:
        continue
    
    for j in range(args.num_samples):
        output_path = os.path.join(individual_eval_inference_output_dir, f'{i}_r{j}.json')
        assert os.path.exists(output_path), f"Expect {output_path} to exist"
        if j == 0:
            inference_eval_examples[i][prediction_key] = []
        
        with open(output_path, 'r') as f:
            current_prediction_data = json.load(f)
        inference_eval_examples[i][prediction_key].append(current_prediction_data['prediction'])

print('Done combining, now evaluating')

# Evaluate predictions
for i in range(len(inference_eval_examples)):
    if i in skip_problems:
        continue
    
    # Get ground truth answer
    solution_or_answer = str(inference_eval_examples[i][answer_key])
    if dataset_type == 'MATH':
        answer_processed = process_sample(solution_or_answer, None, extract_last_occurrence)
    else:
        answer_processed = solution_or_answer
    
    # Process predictions
    all_samples = inference_eval_examples[i][prediction_key]
    all_processed_predictions = [process_sample(sample, None, extract_last_occurrence) for sample in all_samples]
    predictions_correctness = [match_fn(sample, answer_processed) if sample is not None else False 
                              for sample in all_processed_predictions]
    
    # Compute metrics
    inference_eval_examples[i]['predictions_correctness'] = predictions_correctness
    inference_eval_examples[i]['pass_at_k'] = any(predictions_correctness)
    sample_partition = equivalence_partition(all_processed_predictions, match_fn)
    majority_vote_correct = compute_majority_vote_correct(all_processed_predictions, predictions_correctness, 
                                                          sample_partition, strict_tie_breaking=False)
    inference_eval_examples[i]['majority_vote_correct'] = majority_vote_correct

# Save results
write_jsonl(inference_eval_examples, results_path)

# Calculate aggregated accuracy
single_sample_accuracy_list = get_average_reward(inference_eval_examples, 'predictions_correctness', 100)
single_sample_accuracy_mean = np.mean(single_sample_accuracy_list)
majority_vote_accuracy_mean = np.mean([ex['majority_vote_correct'] for ex in inference_eval_examples])
pass_k_accuracy_mean = np.mean([ex['pass_at_k'] for ex in inference_eval_examples])

# Save stats
with open(stats_path, 'w') as f:
    json.dump({
        'single_sample_accuracy_mean': single_sample_accuracy_mean,
        'majority_vote_accuracy_mean': majority_vote_accuracy_mean,
        'pass_k_accuracy_mean': pass_k_accuracy_mean
    }, f, indent=2)

print('Single sample accuracy:', single_sample_accuracy_mean)
print('Majority vote accuracy:', majority_vote_accuracy_mean)
print('Pass@K accuracy:', pass_k_accuracy_mean)
