import argparse
import json
import os
from tqdm import tqdm
import glob
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import copy
from accuracy_utils import sample_match_strict, process_sample, numeric_or_symbolic_correctness, \
    equivalence_partition, compute_majority_vote_correct
from qwen_classifier import CustomModelForSequenceClassification, CustomValueGuidedLogitProcessor
from utils import read_jsonl, tokenize_with_chat_template, generate_with_classifier_guidance, write_jsonl, \
    get_average_reward, get_parent_directory, resolve_dict_value
import utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ref_model_id', default=None, type=str,
                    help='reference model id meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--classifier_type', default=None, type=str, help='whether to train Q (bottlenecked) or V classifier.')
parser.add_argument('--classifier_model_id', default=None, type=str, help='classifier model id (for tokenizer, reuse weights)')
parser.add_argument('--classifier_ckpt_path', required=True, type=str,
                    help='a ckpt path')
parser.add_argument('--inference_mode', default=None, type=str,
                    help='inference mode supported by the classifier. First round does not matter')
parser.add_argument('--loss_type', default=None, type=str, help='loss type for the classifier, unused for evaluation')
parser.add_argument('--use_bias', default=None, type=int,
                    help='whether to use bias for the classification layer, llama 3 does not have bias')
parser.add_argument('--data_path', default=None, type=str, help='path to the data dataset/gsm8k_train.jsonl')
parser.add_argument('--train_eval_save_path', default=None, type=str,
                    help='train eval split dataset/gsm8k_train_eval.json')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--kl_batch_size', default=2, type=int, help='batch size for KL computation')
parser.add_argument('--num_samples', default=8, type=int, help='number of samples per problem')
parser.add_argument('--cd_baseline', default=0, type=int, help='whether to use CD baseline')
parser.add_argument('--use_chat_template', default=None, type=int, help='whether to use chat template for generation')
parser.add_argument('--eta', default=None, type=float,
                    help='eta for the classifier, larger it is, less KL regularization. Unused for expectation inference mode')
parser.add_argument('--top_k', type=int, default=20, help='top k logits to modify, -1 means all logits')
parser.add_argument('--temperature', default=None, type=float, help='temperature for sampling 0.8')
parser.add_argument('--top_p', default=None, type=float, help='top p for sampling 0.9')
parser.add_argument('--max_prompt_length', default=-1, type=int, help='max tokens for prompt, -1 means no limit')
parser.add_argument('--max_new_tokens', default=1024, type=int, help='max tokens for sampling 1024')
parser.add_argument('--dtype', default=None, type=str, help='data type for the model bfloat16')
parser.add_argument('--match_fn_type', default=None, type=str,
                    help='matching function type for evaluation, symbolic or strict; symbolic')
parser.add_argument('--output_dir', default=None, type=str,
                    help='default use classifier_ckpt_path')
parser.add_argument('--force', default=0, type=int, help='force overwrite existing files')
parser.add_argument('--seed', default=47, type=int, help='seed for reproduction')

parser.add_argument('--num_atoms', default=None, type=int, help='number of atoms for mle classifier')
parser.add_argument('--V_min', default=None, type=float, help='V_min for histogram learning')
parser.add_argument('--V_max', default=None, type=float, help='V_max for histogram learning')
parser.add_argument('--shift_reward', default=None, type=float, help='shift reward by value (subtraction)')
parser.add_argument('--scale_reward', default=None, type=float, help='scale reward by value (multiplication)')

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

with open(os.path.join(get_parent_directory(args.classifier_ckpt_path), 'args.json'), 'r') as f:
    training_args_dict = json.load(f)
print(training_args_dict)

ref_model_id = resolve_dict_value(args_dict, training_args_dict, 'ref_model_id')
classifier_type = resolve_dict_value(args_dict, training_args_dict, 'classifier_type')
classifier_model_id = resolve_dict_value(args_dict, training_args_dict, 'classifier_model_id')
classifier_ckpt_path = args.classifier_ckpt_path
dataset_type = resolve_dict_value(args_dict, training_args_dict, 'dataset_type')
inference_mode = resolve_dict_value(args_dict, training_args_dict, 'inference_mode')
loss_type = resolve_dict_value(args_dict, training_args_dict, 'loss_type')
use_bias = bool(resolve_dict_value(args_dict, training_args_dict, 'use_bias'))
data_path = resolve_dict_value(args_dict, training_args_dict, 'data_path', 'original_problems_path')
train_eval_save_path = resolve_dict_value(args_dict, training_args_dict, 'train_eval_save_path')
batch_size = args.batch_size
kl_batch_size = args.kl_batch_size
num_samples = args.num_samples
cd_baseline = args.cd_baseline
use_chat_template = resolve_dict_value(args_dict, training_args_dict, 'use_chat_template')
eta = resolve_dict_value(args_dict, training_args_dict, 'eta')
top_k = resolve_dict_value(args_dict, training_args_dict, 'top_k')
assert eta >= 0
temperature = resolve_dict_value(args_dict, training_args_dict, 'temperature')
top_p = resolve_dict_value(args_dict, training_args_dict, 'top_p')
max_prompt_length = resolve_dict_value(args_dict, training_args_dict, 'max_prompt_length')
max_new_tokens = resolve_dict_value(args_dict, training_args_dict, 'max_new_tokens')
dtype = resolve_dict_value(args_dict, training_args_dict, 'dtype')
match_fn_type = resolve_dict_value(args_dict, training_args_dict, 'match_fn_type')
output_dir = args.output_dir
force = args.force
seed = args.seed
num_atoms = resolve_dict_value(args_dict, training_args_dict, 'num_atoms')
V_min = resolve_dict_value(args_dict, training_args_dict, 'V_min')
V_max = resolve_dict_value(args_dict, training_args_dict, 'V_max')
shift_reward = resolve_dict_value(args_dict, training_args_dict, 'shift_reward')
scale_reward = resolve_dict_value(args_dict, training_args_dict, 'scale_reward')

if output_dir is None:
    output_dir = classifier_ckpt_path
individual_eval_inference_output_dir = os.path.join(output_dir, 'individual_eval_inference_eta_{0}_top_k_{1}_temp_{2}'.format(eta, top_k, temperature))
os.makedirs(individual_eval_inference_output_dir, exist_ok=True)

if force:
    os.system('rm -rf {0}/*'.format(individual_eval_inference_output_dir))

if not force and os.path.exists(os.path.join(output_dir, 'inference_eval_results_eta_{0}_top_k_{1}_temp_{2}.jsonl'.format(eta, top_k, temperature))) \
    and os.path.exists(os.path.join(output_dir, 'reward_stats_eta_{0}_top_k_{1}_temp_{2}.json'.format(eta, top_k, temperature))):
    print('output exists, skipping')
    exit(0)

os.makedirs(output_dir, exist_ok=True)
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
assert len(tokenizer) == len(classifier_tokenizer), "tokenizer vocab size mismatch"

#Qwen related fix because the model has more outputs that tokenizer.
model_loading_kwargs = {}
if dtype == 'bfloat16':
    model_loading_kwargs['torch_dtype'] = torch.bfloat16
temp_model = AutoModelForCausalLM.from_pretrained(classifier_model_id, **model_loading_kwargs, device_map='cpu')
lm_head_parameters = list(temp_model.lm_head.parameters())
assert len(lm_head_parameters) == 1  # only weight
lm_head_parameters = lm_head_parameters[0].data.to(device)
print(lm_head_parameters.shape)
vocab_size = lm_head_parameters.shape[0]
print('vocab size', vocab_size)
del temp_model
torch.cuda.empty_cache()

if tokenizer.pad_token is None:
    assert 'Llama-3' in ref_model_id
    tokenizer.pad_token = tokenizer.added_tokens_decoder[128002].content  # reserved special token 0
tokenizer.padding_side = "left"  # for inference
print('tokenizer padding side:', tokenizer.padding_side)
if temperature == 0:
    do_sample = False
    temperature = 1.0
else:
    do_sample = True

prediction_key = 'predictions'
reward_model = None
if dataset_type == 'gsm8k':
    dataset_type = 'GSM8K'
    answer_key = 'answer'
    reward_key = 'partial_guided_predictions_correctness'
    extract_last_occurrence = True
elif dataset_type == 'math':
    dataset_type = 'MATH'
    answer_key = 'solution'  # require additional processing
    reward_key = 'partial_guided_predictions_correctness'
    extract_last_occurrence = True
else:
    raise ValueError('Unknown dataset name: %s' % dataset_type)

if match_fn_type == 'strict':
    match_fn = sample_match_strict
elif match_fn_type == 'symbolic':
    match_fn = numeric_or_symbolic_correctness
else:
    raise ValueError('Unknown match function type: %s' % match_fn_type)

with open(train_eval_save_path, 'r') as f:
    train_eval_problems_d = json.load(f)
original_id_to_eval_id_d = {}
original_examples = read_jsonl(data_path)
inference_eval_examples = []
for i in range(len(original_examples)):
    if train_eval_problems_d[original_examples[i]['problem']]['split'] == 'eval':
        inference_eval_examples.append(original_examples[i])
        original_id = train_eval_problems_d[original_examples[i]['problem']]['id']
        original_id_to_eval_id_d[original_id] = len(original_id_to_eval_id_d)

#inference_eval_examples = inference_eval_examples[:100]


# disable top_k and temperature
generate_kwargs = {'temperature': temperature, 'top_p': top_p, 'do_sample': do_sample, 'max_new_tokens': max_new_tokens, "top_k": 0}
model_loading_kwargs = {}
if dtype == 'bfloat16':
    model_loading_kwargs['torch_dtype'] = torch.bfloat16
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, **model_loading_kwargs, device_map=device)
classifier_model = CustomModelForSequenceClassification.from_pretrained(classifier_ckpt_path, **model_loading_kwargs,
                                                                        num_labels=vocab_size, classifier_type=classifier_type,
                                                                        loss_type=loss_type, use_bias=use_bias,
                                                                        device_map=device, num_atoms=num_atoms,
                                                                        V_min=V_min, V_max=V_max)

ref_model.eval()
classifier_model.eval()
torch.set_grad_enabled(False)  # disable gradients globally
if eta != 0:
    logit_processor = CustomValueGuidedLogitProcessor(eta=eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
                                                      value_classifier=classifier_model, inference_mode=inference_mode, top_k=top_k, cd_baseline=cd_baseline,
                                                      use_cache=True)
else:
    logit_processor = CustomValueGuidedLogitProcessor(eta=eta, ref_model=ref_model, ref_model_tokenizer=tokenizer,
                                                      value_classifier=classifier_model,
                                                      inference_mode='disabled', top_k=top_k, cd_baseline=cd_baseline,
                                                      use_cache=True)

# skip problems due to prompts being too long
skip_problems = []
for j in range(len(inference_eval_examples)):
    # skip problems that are too long
    num_toks = len(tokenizer(inference_eval_examples[j]['prompt'])['input_ids'])
    if max_prompt_length != -1 and num_toks > max_prompt_length:
        skip_problems.append(j)


for i in range(num_samples):
    repeat_index = i
    current_seed = seed + 50 * repeat_index
    set_seed(current_seed)
    print('repeat {0}'.format(repeat_index))
    if not force:
        existing_data_paths = glob.glob(os.path.join(individual_eval_inference_output_dir, '*_r{0}.json'.format(repeat_index)))
        existing_indices = [int(os.path.basename(path).split('_')[0]) for path in existing_data_paths]
    else:
        existing_indices = []
    data_to_infer = []
    for j in range(len(inference_eval_examples)):
        if j in existing_indices or j in skip_problems:
            continue

        data_to_infer.append(copy.deepcopy(inference_eval_examples[j]))
    print('total number of problems to infer for repeat {0}:'.format(repeat_index), len(data_to_infer))
    num_batches = math.ceil(len(data_to_infer) / batch_size)
    for j in tqdm(range(num_batches)):
        batch_start_index = j * batch_size
        batch_end_index = min((j + 1) * batch_size, len(data_to_infer))
        batch_indices = list(range(batch_start_index, batch_end_index))
        current_prompts = [data_to_infer[k]['prompt'] for k in range(batch_start_index, batch_end_index)]
        current_inputs, current_formatted_prompts = tokenize_with_chat_template(tokenizer, current_prompts,
                                                                                use_chat_template, device)
        generate_kwargs['output_scores'] = True
        generate_kwargs['return_dict_in_generate'] = True
        current_outputs = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, generate_kwargs, True, False)
        current_outputs_id = current_outputs['sequences']
        current_outputs_text = tokenizer.batch_decode(current_outputs_id, skip_special_tokens=True)
        current_outputs['scores'] = tuple([e.cpu() for e in current_outputs['scores']])  # prevent OOM
        aligned_model_scores = torch.stack(current_outputs['scores'], dim=1).float()
        del current_outputs
        torch.cuda.empty_cache()

        # also evaluate the KL divergence w.r.t. ref model
        token_kl_list = []
        for k in range(0, len(batch_indices), kl_batch_size):
            # compute kl in batches since kl computation is memory intensive
            # we want KL(pi_aligned || pi_ref)
            output_attention_mask = (current_outputs_id[k:k + kl_batch_size] != tokenizer.pad_token_id).long()
            concat_input_ids = torch.cat([current_inputs['input_ids'][k:k + kl_batch_size], current_outputs_id[k:k + kl_batch_size]], dim=1)
            concat_attention_mask = torch.cat([current_inputs['attention_mask'][k:k + kl_batch_size], output_attention_mask], dim=1)
            concat_inputs = {'input_ids': concat_input_ids, 'attention_mask': concat_attention_mask}
            ref_model_output = ref_model(**concat_inputs)
            ref_model_output_logits = ref_model_output.logits[:, current_inputs['input_ids'].shape[1] - 1:-1]
            ref_model_output_logits = ref_model_output_logits.float() / temperature
            del ref_model_output
            torch.cuda.empty_cache()

            cur_token_kl = utils.kl_divergence(aligned_model_scores[k:k+kl_batch_size].to(ref_model_output_logits.device), ref_model_output_logits)
            cur_token_kl = cur_token_kl * output_attention_mask
            token_kl_list.append(cur_token_kl)
            torch.cuda.empty_cache()

            del ref_model_output_logits
            torch.cuda.empty_cache()

        token_kl = torch.cat(token_kl_list, dim=0)
        traj_kl = token_kl.sum(dim=1)
        del aligned_model_scores
        torch.cuda.empty_cache()

        # save the results
        for k in range(len(batch_indices)):
            original_problem_id = train_eval_problems_d[data_to_infer[batch_indices[k]]['problem']]['id']
            eval_problem_id = original_id_to_eval_id_d[original_problem_id]
            current_output_path = os.path.join(individual_eval_inference_output_dir, f'{eval_problem_id}_r{repeat_index}.json')
            assert not os.path.exists(current_output_path), f"expect {current_output_path} to not exist"
            with open(current_output_path, 'w') as f:
                json.dump({
                    'input_ids': current_inputs['input_ids'][k].cpu().tolist(),
                    'output_ids': current_outputs_id[k].cpu().tolist(),
                    'prediction': current_outputs_text[k],
                    'token_kl': token_kl[k].cpu().tolist(),
                    'traj_kl': traj_kl[k].item(),
                }, f)

print('done inference, now combine results')



for i in range(len(inference_eval_examples)):
    if i in skip_problems:
        continue

    for j in range(num_samples):
        current_output_path = os.path.join(individual_eval_inference_output_dir, f'{i}_r{j}.json')
        assert os.path.exists(current_output_path), f"expect {current_output_path} to exist"
        if j == 0:
            inference_eval_examples[i][prediction_key] = []
            inference_eval_examples[i]['token_kl'] = []
            inference_eval_examples[i]['traj_kl'] = []

        with open(current_output_path, 'r') as f:
            current_prediction_data = json.load(f)
        inference_eval_examples[i][prediction_key].append(current_prediction_data['prediction'])
        inference_eval_examples[i]['token_kl'].append(current_prediction_data['token_kl'])
        inference_eval_examples[i]['traj_kl'].append(current_prediction_data['traj_kl'])

print('done combining results, now evaluate')




for i in range(len(inference_eval_examples)):
    if i in skip_problems:
        continue
    solution_or_answer = str(inference_eval_examples[i][answer_key])
    if dataset_type == 'MATH':
        # process answers a bit
        answer_processed = process_sample(solution_or_answer, None, extract_last_occurrence)
    else:
        answer_processed = solution_or_answer
    all_samples = inference_eval_examples[i][prediction_key]
    all_processed_predictions = [process_sample(sample, None, extract_last_occurrence) for sample in all_samples]
    predictions_correctness = [match_fn(sample, answer_processed) if sample is not None else False for sample in all_processed_predictions]
    inference_eval_examples[i]['predictions_correctness'] = predictions_correctness
    pass_at_k = any(predictions_correctness)
    sample_partition = equivalence_partition(all_processed_predictions, match_fn)
    majority_vote_correct = compute_majority_vote_correct(all_processed_predictions, predictions_correctness, sample_partition, strict_tie_breaking=False)
    inference_eval_examples[i]['majority_vote_correct'] = majority_vote_correct
    inference_eval_examples[i]['pass_at_k'] = pass_at_k

write_jsonl(inference_eval_examples, os.path.join(output_dir, 'inference_eval_results_eta_{0}_top_k_{1}_temp_{2}.jsonl'.format(eta, top_k, temperature)))

# calculate aggregated accuracy
single_sample_accuracy_list = get_average_reward(inference_eval_examples, 'predictions_correctness', 100)
single_sample_accuracy_mean = np.mean(single_sample_accuracy_list)
majority_vote_accuracy_mean = np.mean([example['majority_vote_correct'] for example in inference_eval_examples])
pass_k_accuracy_mean = np.mean([example['pass_at_k'] for example in inference_eval_examples])

with open(os.path.join(output_dir, 'reward_stats_eta_{0}_top_k_{1}_temp_{2}.json'.format(eta, top_k, temperature)), 'w') as f:
    json.dump({'single_sample_accuracy_mean': single_sample_accuracy_mean,
            'majority_vote_accuracy_mean': majority_vote_accuracy_mean,
            'pass_k_accuracy_mean': pass_k_accuracy_mean}, f)

print('single_sample_accuracy_mean', single_sample_accuracy_mean)
print('majority_vote_accuracy_mean', majority_vote_accuracy_mean)
print('pass_k_accuracy_mean', pass_k_accuracy_mean)
