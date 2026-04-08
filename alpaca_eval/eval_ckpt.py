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
import datasets
from classifier import CustomLlamaForSequenceClassification, CustomValueGuidedLogitProcessor
from utils import read_jsonl, tokenize_with_chat_template, generate_with_classifier_guidance, write_jsonl, \
    get_parent_directory, resolve_dict_value, write_json_array
import utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ref_model_id', default=None, type=str,
                    help='reference model id meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument('--classifier_type', default=None, type=str, help='whether to train Q (bottlenecked) or V classifier.')
parser.add_argument('--classifier_model_id', default=None, type=str, help='classifier model id (for tokenizer, reuse weights)')
parser.add_argument('--classifier_ckpt_path', required=True, type=str,
                    help='a ckpt path')
parser.add_argument('--inference_mode', default=None, type=str,
                    help='inference mode supported by the classifier. First round does not matter')
parser.add_argument('--loss_type', default=None, type=str, help='loss type for the classifier, unused for evaluation')
parser.add_argument('--use_bias', default=None, type=int,
                    help='whether to use bias for the classification layer, llama 3 does not have bias')
parser.add_argument('--data_path', default='tatsu-lab/alpaca_eval', type=str, help='Data to generate outputs for (default is the standard Alpaca-Eval dataset)')
parser.add_argument('--data_split', default="alpaca_eval", type=str, help='Data split to use (default is given for the tatsu-lab/alpaca_eval dataset)')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--kl_batch_size', default=2, type=int, help='batch size for KL computation')
parser.add_argument('--num_samples', default=1, type=int, help='number of samples per problem')
parser.add_argument('--cd_baseline', default=0, type=int, help='whether to use CD baseline')
parser.add_argument('--use_chat_template', default=None, type=int, help='whether to use chat template for generation')
parser.add_argument('--eta', default=None, type=float,
                    help='eta for the classifier, larger it is, less KL regularization. Unused for expectation inference mode')
parser.add_argument('--top_k', type=int, default=20, help='top k logits to modify, -1 means all logits')
parser.add_argument('--temperature', default=None, type=float, help='temperature for sampling 0.8')
parser.add_argument('--top_p', default=None, type=float, help='top p for sampling 0.9')
parser.add_argument('--max_prompt_length', default=-1, type=int, help='max tokens for prompt, -1 means no limit')
parser.add_argument('--max_new_tokens', default=4096, type=int, help='max tokens for sampling 4096')
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
parser.add_argument('--quick_test', action='store_true', help='whether to run a quick test with 10 samples for debugging purposes')

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
data_path = args.data_path
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
individual_eval_inference_output_dir = os.path.join(output_dir, 'individual_eval_inference_{3}_eta_{0}_top_k_{1}_temp_{2}'.format(eta, top_k, temperature, args.seed))
os.makedirs(individual_eval_inference_output_dir, exist_ok=True)

if force:
    os.system('rm -rf {0}/*'.format(individual_eval_inference_output_dir))

if not force and os.path.exists(os.path.join(output_dir, 'inference_eval_results_{3}_eta_{0}_top_k_{1}_temp_{2}.jsonl'.format(eta, top_k, temperature, args.seed))) \
    and os.path.exists(os.path.join(output_dir, 'reward_stats_{3}_eta_{0}_top_k_{1}_temp_{2}.json'.format(eta, top_k, temperature, args.seed))):
    print('output exists, skipping')
    exit(0)

os.makedirs(output_dir, exist_ok=True)
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
assert len(tokenizer) == len(classifier_tokenizer), "tokenizer vocab size mismatch"
vocab_size = len(tokenizer)
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

reward_model = None

# disable top_k and temperature
generate_kwargs = {'temperature': temperature, 'top_p': top_p, 'do_sample': do_sample, 'max_new_tokens': max_new_tokens, "top_k": 0}
model_loading_kwargs = {}
if dtype == 'bfloat16':
    model_loading_kwargs['torch_dtype'] = torch.bfloat16
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, **model_loading_kwargs, device_map=device)
classifier_model = CustomLlamaForSequenceClassification.from_pretrained(classifier_ckpt_path, **model_loading_kwargs,
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

inference_eval_examples = datasets.load_dataset(args.data_path, args.data_split, trust_remote_code=True)["eval"]
if args.quick_test:
    inference_eval_examples = inference_eval_examples.select(range(10))
# Remove the existing output column if it exists to avoid confusion with new outputs
inference_eval_examples.remove_columns("output")
# Convert to list of dicts since that matches the logic used later in the code
inference_eval_examples = inference_eval_examples.to_list()
example_to_ID_map = {example['instruction']: indx for indx, example in enumerate(inference_eval_examples)}

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
        if j in existing_indices:
            continue
        data_to_infer.append(copy.deepcopy(inference_eval_examples[j]))

    print('total number of problems to infer for repeat {0}:'.format(repeat_index), len(data_to_infer))
    num_batches = math.ceil(len(data_to_infer) / batch_size)
    for j in tqdm(range(num_batches)):
        batch_start_index = j * batch_size
        batch_end_index = min((j + 1) * batch_size, len(data_to_infer))
        batch_indices = list(range(batch_start_index, batch_end_index))
        current_prompts = [data_to_infer[k]['instruction'] for k in range(batch_start_index, batch_end_index)]
        # I am using the same chat template as in math_reasoning, since it is very generic
        current_inputs, current_formatted_prompts = tokenize_with_chat_template(tokenizer, current_prompts,
                                                                                use_chat_template, device)
        generate_kwargs['output_scores'] = True
        generate_kwargs['return_dict_in_generate'] = True
        current_outputs = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, generate_kwargs, True, False, eta)
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
            problem_id = example_to_ID_map[current_prompts[k]]
            current_output_path = os.path.join(individual_eval_inference_output_dir, f'{problem_id}_r{repeat_index}.json')
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
    for j in range(num_samples):
        current_output_path = os.path.join(individual_eval_inference_output_dir, f'{i}_r{j}.json')
        assert os.path.exists(current_output_path), f"expect {current_output_path} to exist"
        if j == 0:
            inference_eval_examples[i]['token_kl'] = []
            inference_eval_examples[i]['traj_kl'] = []
            inference_eval_examples[i]['output'] = []

        with open(current_output_path, 'r') as f:
            current_prediction_data = json.load(f)
        inference_eval_examples[i]['token_kl'].append(current_prediction_data['token_kl'])
        inference_eval_examples[i]['traj_kl'].append(current_prediction_data['traj_kl'])
        inference_eval_examples[i]['output'] = current_prediction_data['prediction']

print('done combining results, now saving')

# Save this file just like in math_reasoning so the KL can be computed
write_jsonl(inference_eval_examples, os.path.join(output_dir, 'inference_eval_results_{3}_eta_{0}_top_k_{1}_temp_{2}.jsonl'.format(eta, top_k, temperature, args.seed)))

# Save model_outputs.jsonl
model_outputs = []
for i in range(len(inference_eval_examples)):
    model_outputs.append({
        'dataset': inference_eval_examples[i]['dataset'],
        'instruction': inference_eval_examples[i]['instruction'],
        'output': inference_eval_examples[i]['output'],
        'generator': args.classifier_ckpt_path.strip('/').split('/')[-1]
    })
write_json_array(model_outputs, os.path.join(output_dir, 'model_outputs.json'))