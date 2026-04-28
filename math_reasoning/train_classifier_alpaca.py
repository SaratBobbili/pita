import argparse
import json
import os
import socket
import time

import numpy as np

import torch
import wandb
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, DataCollatorForLanguageModeling, \
    get_constant_schedule_with_warmup

import utils
from accuracy_utils import sample_match_strict, numeric_or_symbolic_correctness
from classifier import CustomLlamaForSequenceClassification
from utils import read_jsonl, create_classifier_data, CustomClassifierDataset, calculate_explained_variance, \
    calculate_r2, DynamicBatchSampler, calculate_mle_stats, custom_collate_fn
from functools import partial

parser = argparse.ArgumentParser(description='')
parser.add_argument('--world_size', default=1, type=int, help='number of processes / gpus')
parser.add_argument('--ref_model_id', default='meta-llama/Meta-Llama-3-8B-Instruct', type=str,
                    help='reference model id')
parser.add_argument('--classifier_type', default='V', type=str,
                    help='whether to train Q (bottlenecked) or V classifier.')
parser.add_argument('--classifier_model_id', default='meta-llama/Llama-3.2-1B-Instruct', type=str,
                    help='classifier model id (for tokenizer, reuse weights)')
parser.add_argument('--classifier_ckpt_path', default=None, type=str,
                    help='classifier ckpt path assuming we are loading the full model in, if None, we will load the model from classifier_model_id')
parser.add_argument('--resume_opt_scheduler', default=None, type=int,
                    help='whether to resume optimizer and scheduler from the checkpoint, 0 (no), 1 (yes)')
parser.add_argument('--original_problems_path', required=True, type=str, help='for inference eval')
parser.add_argument('--train_eval_save_path', required=True, type=str, help='')
parser.add_argument('--init_mode', required=True, type=str,
                    help='zero / random / reuse / warmstart init the output layer for the classifier. For second round or higher, warmstart is reusing the previous ckpt without modifying any weight.')
parser.add_argument('--inference_mode', required=True, type=str,
                    help='inference mode supported by the classifier. First round does not matter')
parser.add_argument('--loss_type', default='bce', type=str, help='loss type for the classifier, bce or mse')
parser.add_argument('--use_bias', default=0, type=int,
                    help='whether to use bias for the classification layer, llama 3 does not have bias')
parser.add_argument('--dataset_type', required=True, type=str, help='gsm8k')
parser.add_argument('--data_paths', required=True, nargs='+', type=str, help='all paths to the training data')
parser.add_argument('--batch_size', default=8, type=int, help='batch size for training classifier (max allowed)')
parser.add_argument('--max_batch_num_tokens', default=-1, type=int,
                    help='max number of tokens for each batch, -1 means no limit')
parser.add_argument('--gradient_accumulation_step', default=1, type=int, help='gradient accumulation step')
parser.add_argument('--shift_reward', default=0, type=float, help='shift reward by value (subtraction)')
parser.add_argument('--scale_reward', default=1, type=float, help='scale reward by value (multiplication)')
parser.add_argument('--cd_baseline', default=0, type=int, help='if 1, run the CD baseline.')
parser.add_argument('--use_chat_template', default=1, type=int, help='whether to use chat template for generation')
parser.add_argument('--dtype', default='bfloat16', type=str, help='data type for the model bfloat16 or empty string')
parser.add_argument('--temperature', default=0.8, type=float, help='temperature for sampling')
parser.add_argument('--use_all_ref_tokens', default=1, type=int,
                    help='whether to use all tokens from the reference model for training 0 (no), 1 (yes, random cut still applies), 2 (everything, should only be used for the first round of training)')
parser.add_argument('--top_p', default=0.9, type=float, help='top p for sampling')
parser.add_argument('--drop_no_variation', default=1, type=int,
                    help='whether to drop problems with no variation in the correctness label')
parser.add_argument('--id_eval_ratio', default=0.1, type=float, help='ratio of id samples for evaluation')
parser.add_argument('--eta', default=None, type=float,
                    help='eta for the classifier, larger it is, less KL regularization')
parser.add_argument('--top_k', type=int, default=-1, help='top k logits to modify, -1 means all logits')
parser.add_argument('--match_fn_type', default='symbolic', type=str,
                    help='matching function type for evaluation, symbolic or strict')
parser.add_argument('--output_dir', default='checkpoints/temp/', type=str, help='checkpoints/exp1/round_0/')
parser.add_argument('--num_workers', default=0, type=int,
                    help='number of workers for data loader; values other than 0 could cause issues with tokenizer')
parser.add_argument('--num_epochs', default=3, type=int, help='number of epochs for training')
parser.add_argument('--eval_max_size', default=1000, type=int, help='number of epochs for training')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate for the classifier')
parser.add_argument('--warmup_step', default=-1, type=int, help='warmup steps for the classifier, -1 means no warmup')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay for the classifier')
parser.add_argument('--eval_freq', default=500, type=int, help='evaluation frequency')
parser.add_argument('--ckpt_freq', default=5000, type=int, help='checkpoint frequency')
parser.add_argument('--save_opt_scheduler', default=0, type=int, help='whether to save optimizer and scheduler state')
parser.add_argument('--seed', default=47, type=int, help='seed for reproduction')
parser.add_argument('--track', default=0, type=int, help='whether to report to wandb')
parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity')
parser.add_argument('--wandb_project', default="", type=str, help='wandb project')
parser.add_argument('--wandb_run_name', default="", type=str, help='wandb run name')
parser.add_argument('--num_atoms', default=11, type=int, help='number of atoms for mle classifier')
parser.add_argument('--V_min', default=0, type=float, help='V_min for histogram learning')
parser.add_argument('--V_max', default=1, type=float, help='V_max for histogram learning')
parser.add_argument('--max_length', default=-1, type=int, help='max tokens for training')
parser.add_argument('--save_raw_efficiency', default=1, type=int,
                    help='whether to save raw efficiency logs')
parser.add_argument('--efficiency_log_every', default=1, type=int,
                    help='log efficiency every N optimizer steps')
parser.add_argument('--sync_cuda_timing', default=1, type=int,
                    help='synchronize CUDA before/after timing')
parser.add_argument('--efficiency_log_dir', default=None, type=str,
                    help='directory for efficiency logs; defaults to output_dir/efficiency')

args = parser.parse_args()
print(socket.gethostname())
print(vars(args))

world_size = args.world_size
ref_model_id = args.ref_model_id
classifier_model_id = args.classifier_model_id
classifier_ckpt_path = args.classifier_ckpt_path
resume_opt_scheduler = args.resume_opt_scheduler
original_problems_path = args.original_problems_path
train_eval_save_path = args.train_eval_save_path
init_mode = args.init_mode
inference_mode = args.inference_mode
loss_type = args.loss_type
use_bias = bool(args.use_bias)
dataset_type = args.dataset_type
data_paths = args.data_paths
batch_size = args.batch_size
max_batch_num_tokens = args.max_batch_num_tokens
gradient_accumulation_step = args.gradient_accumulation_step
shift_reward = args.shift_reward
scale_reward = args.scale_reward
use_chat_template = args.use_chat_template
dtype = args.dtype
use_all_ref_tokens = args.use_all_ref_tokens
temperature = args.temperature
top_p = args.top_p
drop_no_variation = bool(args.drop_no_variation)
id_eval_ratio = args.id_eval_ratio
eta = args.eta
top_k = args.top_k
match_fn_type = args.match_fn_type
output_dir = args.output_dir
num_workers = args.num_workers
num_epochs = args.num_epochs
eval_max_size = args.eval_max_size
lr = args.lr
warmup_step = args.warmup_step
weight_decay = args.weight_decay
eval_freq = args.eval_freq
ckpt_freq = args.ckpt_freq
save_opt_scheduler = bool(args.save_opt_scheduler)
seed = args.seed
save_raw_efficiency = bool(args.save_raw_efficiency)
efficiency_log_every = args.efficiency_log_every
sync_cuda_timing = bool(args.sync_cuda_timing)
efficiency_log_dir = args.efficiency_log_dir
assert efficiency_log_every >= 1, 'efficiency_log_every must be >= 1'

if classifier_ckpt_path is None:
    classifier_ckpt_path = classifier_model_id
else:
    assert resume_opt_scheduler is not None, 'resume_opt_scheduler must be specified when classifier_ckpt_path is not None'

accelerator = Accelerator()
set_seed(seed * 42 + accelerator.process_index * 100003)  # prime

if accelerator.is_main_process:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
if efficiency_log_dir is None:
    efficiency_log_dir = os.path.join(output_dir, 'efficiency')
if accelerator.is_main_process and save_raw_efficiency:
    utils.ensure_dir(efficiency_log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
assert len(tokenizer) == len(classifier_tokenizer), "tokenizer vocab size mismatch"
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
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
if dataset_type == 'gsm8k':
    dataset_type = 'GSM8K'
    answer_key = 'answer'
    reward_key = 'partial_guided_predictions_correctness'
elif dataset_type == 'math':
    dataset_type = 'MATH'
    answer_key = 'solution'  # require additional processing
    reward_key = 'partial_guided_predictions_correctness'
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

# group all data_paths data
problem_position_d = {}
all_data = []
merge_keys = ['fully_guided_predictions', 'fully_guided_predictions_correctness', 'partial_guided_prompts',
              'partial_guided_prompts_tokenized', 'num_response_tokens_in_partial_guided_prompts',
              'partial_guided_responses_tokenized', 'partial_guided_predictions',
              'partial_guided_predictions_correctness']
for data_path in data_paths:
    current_data = read_jsonl(data_path)
    for i in range(len(current_data)):
        current_problem = current_data[i]['problem']
        if current_problem not in problem_position_d:
            problem_position_d[current_problem] = len(all_data)
            all_data.append(current_data[i])
        else:
            for k in merge_keys:
                all_data[problem_position_d[current_problem]][k].extend(current_data[i][k])

# shift and calculate reward
# if inference mode is expectation, we calculate the exp with eta; else no need to do so
num_below_V_min = 0
num_above_V_max = 0
num_rewards = 0
for i in range(len(all_data)):
    current_data = all_data[i]
    assert 'reward' not in current_data
    current_rewards = []
    for j in range(len(current_data[reward_key])):
        current_reward = int(current_data[reward_key][j])
        # print('current reward from top', current_reward)
        # current_reward -= shift_reward
        # if inference_mode == 'expectation' and not args.cd_baseline:
        #     current_reward = np.exp(eta * current_reward)
        #     print('current reward', current_reward)
        # assert 0 <= current_reward <= 1
        current_rewards.append(current_reward)
    current_data['reward'] = current_rewards

if loss_type == "bce":
    all_rewards = list(x['reward'][0] for x in all_data)
    all_rewards = np.array(all_rewards)
    min_reward, max_reward = np.quantile(all_rewards, [0, 1])
    assert min_reward >= 0 and max_reward <= 1, f"min reward: {min_reward}, max reward: {max_reward} should be in [0, 1] for bce loss."

all_train_data = []
all_eval_data = []
for i in range(len(all_data)):
    if train_eval_problems_d[all_data[i]['problem']]['split'] == 'train':
        all_train_data.append(all_data[i])
    elif train_eval_problems_d[all_data[i]['problem']]['split'] == 'eval':
        all_eval_data.append(all_data[i])
    else:
        raise ValueError('Unknown split: %s' % train_eval_problems_d[all_data[i]['problem']]['split'])

print('total number of training problems', len(all_train_data))
print('total number of eval problems', len(all_eval_data))

all_train_classifier_data = create_classifier_data(all_train_data, use_all_ref_tokens, drop_no_variation,
                                                   args.max_length)
all_train_length = len(all_train_classifier_data['input_ids'])
shuffled_indices = np.random.choice(all_train_length, all_train_length, replace=False)
train_indices = shuffled_indices[:int(all_train_length * (1 - id_eval_ratio))]
id_eval_indices = shuffled_indices[int(all_train_length * (1 - id_eval_ratio)):]
train_classifier_data = {k: [all_train_classifier_data[k][i] for i in train_indices] for k in all_train_classifier_data}
id_eval_classifier_data = {k: [all_train_classifier_data[k][i] for i in id_eval_indices] for k in
                           all_train_classifier_data}
ood_eval_classifier_data = create_classifier_data(all_eval_data, 1, False,
                                                  args.max_length)  # prevent evaluation being biased by favoring the later tokens

if eval_max_size != -1:
    if eval_max_size < len(id_eval_classifier_data['input_ids']):
        id_eval_random_indices = np.random.choice(len(id_eval_classifier_data['input_ids']), eval_max_size,
                                                  replace=False)
        id_eval_classifier_data = {k: [id_eval_classifier_data[k][i] for i in id_eval_random_indices] for k in
                                   id_eval_classifier_data}
        print('id_eval indices sum', np.sum(id_eval_random_indices))
    if eval_max_size < len(ood_eval_classifier_data['input_ids']):
        ood_eval_random_indices = np.random.choice(len(ood_eval_classifier_data['input_ids']), eval_max_size,
                                                   replace=False)
        ood_eval_classifier_data = {k: [ood_eval_classifier_data[k][i] for i in ood_eval_random_indices] for k in
                                    ood_eval_classifier_data}
        print('ood_eval indices sum', np.sum(ood_eval_random_indices))

print('total number of training samples', len(train_classifier_data['input_ids']))
print('total number of id eval samples', len(id_eval_classifier_data['input_ids']))
print('total number of ood eval samples', len(ood_eval_classifier_data['input_ids']))

train_classifier_dataset = CustomClassifierDataset(train_classifier_data)
id_eval_classifier_dataset = CustomClassifierDataset(id_eval_classifier_data)
ood_eval_classifier_dataset = CustomClassifierDataset(ood_eval_classifier_data)
custom_collate_fn = partial(custom_collate_fn, pad_token_id=tokenizer.pad_token_id)
if max_batch_num_tokens == -1:
    train_classifier_loader = DataLoader(train_classifier_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True)
    id_eval_classifier_loader = DataLoader(id_eval_classifier_dataset, batch_size=batch_size, shuffle=False,
                                           drop_last=True, num_workers=num_workers, collate_fn=custom_collate_fn,
                                           pin_memory=True)
    ood_eval_classifier_loader = DataLoader(ood_eval_classifier_dataset, batch_size=batch_size, shuffle=False,
                                            drop_last=True, num_workers=num_workers, collate_fn=custom_collate_fn,
                                            pin_memory=True)
    print("Finished creating vanilla dataloader")
else:
    train_sampler = DynamicBatchSampler(train_classifier_dataset, batch_size, max_batch_num_tokens, shuffle=True)
    id_eval_sampler = DynamicBatchSampler(id_eval_classifier_dataset, batch_size, max_batch_num_tokens, shuffle=False)
    ood_eval_sampler = DynamicBatchSampler(ood_eval_classifier_dataset, batch_size, max_batch_num_tokens, shuffle=False)
    train_classifier_loader = DataLoader(train_classifier_dataset, batch_sampler=train_sampler, num_workers=num_workers,
                                         collate_fn=custom_collate_fn, pin_memory=True)
    id_eval_classifier_loader = DataLoader(id_eval_classifier_dataset, batch_sampler=id_eval_sampler,
                                           num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True)
    ood_eval_classifier_loader = DataLoader(ood_eval_classifier_dataset, batch_sampler=ood_eval_sampler,
                                            num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True)
    print("Finished creating dynamic batch dataloader")

model_loading_kwargs = {}
if dtype == 'bfloat16':
    model_loading_kwargs['torch_dtype'] = torch.bfloat16
classifier_model = CustomLlamaForSequenceClassification.from_pretrained(classifier_ckpt_path, **model_loading_kwargs,
                                                                        num_labels=vocab_size,
                                                                        loss_type=loss_type, use_bias=use_bias,
                                                                        classifier_type=args.classifier_type,
                                                                        device_map=device, num_atoms=args.num_atoms,
                                                                        V_min=args.V_min, V_max=args.V_max)
print("Loaded classifier model")
num_trainable_params = utils.count_parameters(classifier_model, trainable_only=True)
num_total_params = utils.count_parameters(classifier_model, trainable_only=False)

if args.classifier_type == 'Q':
    if init_mode == 'zero':
        print('before loading score weight mean', classifier_model.score.weight.data.mean().item())
        classifier_model.zero_init_classifier()
        print('after loading score weight mean', classifier_model.score.weight.data.mean().item())
    elif init_mode == 'reuse':
        temp_model = AutoModelForCausalLM.from_pretrained(classifier_model_id, **model_loading_kwargs, device_map='cpu')
        lm_head_parameters = list(temp_model.lm_head.parameters())
        assert len(lm_head_parameters) == 1  # only weight
        print('before loading score weight mean', classifier_model.score.weight.data.mean().item())
        print('original lm_head weight mean', lm_head_parameters[0].data.mean().item())
        lm_head_parameters = lm_head_parameters[0].data.to(device)
        vocab_size = lm_head_parameters.shape[0]
        if loss_type in ["mle", "qr"]:
            classifier_model.score.weight.data = lm_head_parameters.repeat(1, args.num_atoms).view(
                vocab_size * args.num_atoms, -1)
        else:
            classifier_model.score.weight.data = lm_head_parameters
        del temp_model
        torch.cuda.empty_cache()
        print('after loading score weight mean', classifier_model.score.weight.data.mean().item())
    else:
        assert init_mode == 'random' or init_mode == "warmstart"

optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
if resume_opt_scheduler == 1:
    optimizer_scheduler_state = torch.load(os.path.join(classifier_ckpt_path, 'optimizer_scheduler_state.pt'))
    optimizer.load_state_dict(optimizer_scheduler_state['optimizer'])
    scheduler.load_state_dict(optimizer_scheduler_state['scheduler'])
    print('optimizer and scheduler resumed from the checkpoint')

global_step = 0
optimizer_step_idx = 0
start_time = time.time()
accumulated_loss = torch.tensor(0.0).to(device)
train_step_metrics_path = os.path.join(efficiency_log_dir, 'train_step_metrics.jsonl')
train_eval_metrics_path = os.path.join(efficiency_log_dir, 'train_eval_metrics.jsonl')
run_metadata_path = os.path.join(efficiency_log_dir, 'run_metadata.json')
total_train_examples = 0.0
total_train_tokens = 0.0
total_train_loss_tokens = 0.0
total_train_flops_trainable = 0.0
total_train_flops_total = 0.0

if accelerator.is_main_process and save_raw_efficiency:
    with open(run_metadata_path, 'w') as f:
        json.dump({
            'script': 'train_classifier.py',
            'world_size': accelerator.num_processes,
            'num_trainable_params': int(num_trainable_params),
            'num_total_params': int(num_total_params),
            'ref_model_id': ref_model_id,
            'classifier_model_id': classifier_model_id,
            'dtype': dtype,
            'start_time_unix': start_time,
        }, f, indent=2)

if args.track and accelerator.is_local_main_process:
    from datetime import datetime

    current_date = datetime.now()
    date_string = current_date.strftime('%Y-%m-%d')
    wandb_kwargs = {'entity': args.wandb_entity, 'project': args.wandb_project, 'name': args.wandb_run_name,
                    'config': vars(args), 'tags': [date_string, dataset_type]}
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    run = wandb.init(**wandb_kwargs)
else:
    class DummyRun:
        def log(self, *args, **kwargs):
            pass
    run = DummyRun()

# save at the beginning for the purpose of debugging resume training
classifier_model.eval()
save_dir = os.path.join(output_dir, 'ckpt_%d' % global_step)
optimizer_to_save = None
scheduler_to_save = None
if save_opt_scheduler:
    optimizer_to_save = optimizer
    scheduler_to_save = scheduler
utils.save_model(classifier_model, tokenizer, optimizer_to_save, scheduler_to_save, accelerator, save_dir=save_dir,
                 push_to_hub=False)

# training loop
classifier_model.train()

classifier_model, optimizer, train_classifier_loader, id_eval_classifier_loader, ood_eval_classifier_loader, scheduler = \
    accelerator.prepare(classifier_model, optimizer, train_classifier_loader, id_eval_classifier_loader,
                        ood_eval_classifier_loader, scheduler)
for epoch in range(num_epochs):
    run.log({'Epoch': epoch}, step=global_step)
    bar = tqdm(train_classifier_loader) if accelerator.is_local_main_process else train_classifier_loader
    torch.cuda.empty_cache()
    accum_examples_local = 0.0
    accum_tokens_local = 0.0
    accum_loss_tokens_local = 0.0
    accum_flops_trainable_local = 0.0
    accum_flops_total_local = 0.0
    optimizer_step_start_time = None
    for batch_input_data in bar:
        if optimizer_step_start_time is None:
            utils.sync_cuda(sync_cuda_timing)
            optimizer_step_start_time = time.time()
        global_step += world_size
        batch_examples_local = float(batch_input_data['input_ids'].shape[0])
        batch_tokens_local = float(batch_input_data['attention_mask'].sum().item())
        batch_loss_tokens_local = float(batch_input_data['loss_mask'].sum().item())
        accum_examples_local += batch_examples_local
        accum_tokens_local += batch_tokens_local
        accum_loss_tokens_local += batch_loss_tokens_local
        accum_flops_trainable_local += 6.0 * num_trainable_params * batch_tokens_local
        accum_flops_total_local += 6.0 * num_total_params * batch_tokens_local
        # print(batch_input_data['input_ids'].shape, torch.prod(torch.tensor(batch_input_data['input_ids'].shape)).item())
        outputs = classifier_model(input_ids=batch_input_data['input_ids'],
                                   attention_mask=batch_input_data['attention_mask'],
                                   labels=batch_input_data['rewards'], loss_mask=batch_input_data['loss_mask'],
                                   loss_weights=batch_input_data['loss_weights'])
        loss = outputs.loss / gradient_accumulation_step  # normalize for grad accumulation
        del batch_input_data, outputs
        torch.cuda.empty_cache()

        accelerator.backward(loss)
        accumulated_loss += loss.detach()

        # Logging purposes
        if (global_step // world_size + 1) % gradient_accumulation_step == 0:
            if args.classifier_type == "Q":
                grad_norm = torch.tensor(0.0).to(device)
            else:
                grad_norm = accelerator.clip_grad_norm_(classifier_model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            utils.sync_cuda(sync_cuda_timing)
            optimizer_step_wall_sec = time.time() - optimizer_step_start_time

            elapsed_time = time.time() - start_time
            run.log({
                'Training Loss': accelerator.gather(accumulated_loss).mean(),
                'Learning Rate': scheduler.get_last_lr()[0],
                'Steps per Min': global_step / (elapsed_time / 60),
                'Gradient Norm': accelerator.gather(grad_norm).mean(),
            }, step=global_step)
            optimizer_step_idx += 1
            scalar_device = loss.device
            global_examples = accelerator.gather(
                torch.tensor([accum_examples_local], device=scalar_device, dtype=torch.float64)).sum().item()
            global_tokens = accelerator.gather(
                torch.tensor([accum_tokens_local], device=scalar_device, dtype=torch.float64)).sum().item()
            global_loss_tokens = accelerator.gather(
                torch.tensor([accum_loss_tokens_local], device=scalar_device, dtype=torch.float64)).sum().item()
            global_flops_trainable = accelerator.gather(
                torch.tensor([accum_flops_trainable_local], device=scalar_device, dtype=torch.float64)).sum().item()
            global_flops_total = accelerator.gather(
                torch.tensor([accum_flops_total_local], device=scalar_device, dtype=torch.float64)).sum().item()
            if accelerator.is_main_process:
                total_train_examples += global_examples
                total_train_tokens += global_tokens
                total_train_loss_tokens += global_loss_tokens
                total_train_flops_trainable += global_flops_trainable
                total_train_flops_total += global_flops_total
                if save_raw_efficiency and optimizer_step_idx % efficiency_log_every == 0:
                    utils.append_jsonl(train_step_metrics_path, {
                        'global_step': int(global_step),
                        'optimizer_step': int(optimizer_step_idx),
                        'epoch': int(epoch),
                        'step_wall_sec': float(optimizer_step_wall_sec),
                        'num_examples': float(global_examples),
                        'num_tokens': float(global_tokens),
                        'num_loss_tokens': float(global_loss_tokens),
                        'flops_trainable_est': float(global_flops_trainable),
                        'flops_total_est': float(global_flops_total),
                        'examples_per_sec': float(global_examples / max(optimizer_step_wall_sec, 1e-12)),
                        'tokens_per_sec': float(global_tokens / max(optimizer_step_wall_sec, 1e-12)),
                        'loss_tokens_per_sec': float(global_loss_tokens / max(optimizer_step_wall_sec, 1e-12)),
                        'learning_rate': float(scheduler.get_last_lr()[0]),
                        'gradient_norm': float(accelerator.gather(grad_norm).mean().item()),
                        'train_loss_accumulated': float(accelerator.gather(accumulated_loss).mean().item()),
                    })
            accumulated_loss = torch.tensor(0.0).to(device)
            accum_examples_local = 0.0
            accum_tokens_local = 0.0
            accum_loss_tokens_local = 0.0
            accum_flops_trainable_local = 0.0
            accum_flops_total_local = 0.0
            optimizer_step_start_time = None

        if eval_freq != -1 and (global_step % eval_freq == 0 or global_step == 1):
            utils.sync_cuda(sync_cuda_timing)
            eval_start_time = time.time()
            classifier_model.eval()
            unwrapped_classifier_model = classifier_model.module if hasattr(classifier_model,
                                                                            "module") else classifier_model
            eval_losses = {'id': [], 'ood': []}
            eval_predictions = {'id': [], 'ood': []}
            eval_labels = {'id': [], 'ood': []}
            eval_stats = {'id': [], 'ood': []}
            eval_counts = {'id': {'examples': 0.0, 'tokens': 0.0, 'loss_tokens': 0.0},
                           'ood': {'examples': 0.0, 'tokens': 0.0, 'loss_tokens': 0.0}}
            with torch.no_grad():
                def mask_stats(x, mask):
                    # mask has shape [bs, seqlen]
                    if args.classifier_type == "Q":
                        # Q classifier is shifted
                        # x: [bs, seqlen-1]
                        x[~mask[:, 1:]] = -1
                    else:
                        x[~mask] = -1
                    return x

                def flatten_and_remove_masked_elems(x):
                    x = x.flatten()
                    return x[x != -1]
                # ID eval
                for eval_key, eval_loader in [('id', id_eval_classifier_loader), ('ood', ood_eval_classifier_loader)]:
                    for batch_input_data in eval_loader:
                        eval_counts[eval_key]['examples'] += float(batch_input_data['input_ids'].shape[0])
                        eval_counts[eval_key]['tokens'] += float(batch_input_data['attention_mask'].sum().item())
                        eval_counts[eval_key]['loss_tokens'] += float(batch_input_data['loss_mask'].sum().item())
                        # always use ones during eval
                        batch_input_data['loss_weights'] = torch.ones_like(batch_input_data['loss_weights'])
                        outputs = classifier_model(input_ids=batch_input_data['input_ids'],
                                                   attention_mask=batch_input_data['attention_mask'],
                                                   labels=batch_input_data['rewards'],
                                                   loss_mask=batch_input_data['loss_mask'],
                                                   loss_weights=batch_input_data['loss_weights'])
                        loss = outputs.loss
                        eval_losses[eval_key].append(loss)
                        cur_predictions = unwrapped_classifier_model.calculate_predictions(outputs.logits)
                        cur_predictions = mask_stats(cur_predictions, batch_input_data['loss_mask'])
                        eval_predictions[eval_key].append(cur_predictions)
                        # repeat the rewards to shape [bs, seqlen] for purpose of calculating explained var, r2
                        seqlen = cur_predictions.shape[1]
                        cur_labels = batch_input_data['rewards'].unsqueeze(1).repeat(1, seqlen)
                        cur_labels = mask_stats(cur_labels, batch_input_data['loss_mask'])
                        eval_labels[eval_key].append(cur_labels)
                        if loss_type == "mle":
                            mle_stats = calculate_mle_stats(outputs.logits,
                                                            unwrapped_classifier_model.atoms)  # mapping to [bs, seqlen]
                            for mle_key in mle_stats:
                                mle_stats[mle_key] = mask_stats(mle_stats[mle_key], batch_input_data['loss_mask'])
                            eval_stats[eval_key].append(mle_stats)

                        del batch_input_data, outputs
                        torch.cuda.empty_cache()

            eval_losses = {k: torch.mean(accelerator.gather(torch.tensor(v))) for k, v in eval_losses.items()}
            # each eval_prediction is a list of (bs, seqlen) tensors
            flattened_predictions = {}
            flattened_labels = {}
            flattened_stats = {}
            for k in eval_predictions:
                flattened_predictions[k] = torch.cat([flatten_and_remove_masked_elems(v) for v in eval_predictions[k]])
                flattened_labels[k] = torch.cat([flatten_and_remove_masked_elems(v) for v in eval_labels[k]])
                if loss_type == "mle":
                    flattened_stats[k] = {
                        stat: torch.cat([flatten_and_remove_masked_elems(x[stat]) for x in eval_stats[k]]) for stat in
                        eval_stats[k][0].keys()}
            eval_predictions = flattened_predictions
            eval_labels = flattened_labels
            eval_stats = flattened_stats
            run.log({'ID Eval Loss': eval_losses['id'],
                     'ID Eval Explained Variance': calculate_explained_variance(eval_predictions['id'],
                                                                                eval_labels['id']),
                     'ID Eval R^2': calculate_r2(eval_predictions['id'], eval_labels['id']),
                     'ID Eval Prediction Min': torch.min(eval_predictions['id']),
                     'ID Eval Prediction Max': torch.max(eval_predictions['id']),
                     'ID Eval Prediction Mean': torch.mean(eval_predictions['id']),
                     'OOD Eval Loss': eval_losses['ood'],
                     'OOD Eval Explained Variance': calculate_explained_variance(eval_predictions['ood'],
                                                                                 eval_labels['ood']),
                     'OOD Eval R^2': calculate_r2(eval_predictions['ood'], eval_labels['ood']),
                     'OOD Eval Prediction Min': torch.min(eval_predictions['ood']),
                     'OOD Eval Prediction Max': torch.max(eval_predictions['ood']),
                     'OOD Eval Prediction Mean': torch.mean(eval_predictions['ood'])}, step=global_step)
            if loss_type == "mle":
                for k in ['id', 'ood']:
                    eval_stats[k] = {stat: torch.mean(eval_stats[k][stat]) for stat in eval_stats[k]}
                run.log({'ID Eval MLE Stats': eval_stats['id'], 'OOD Eval MLE Stats': eval_stats['ood']},
                        step=global_step)

            if inference_mode == 'bernoulli':
                # for bernoulli, we can calculate the accuracy
                id_eval_rounded_predictions = [float(eval_predictions['id'][i] > 0.5) for i in
                                               range(len(eval_predictions['id']))]
                ood_eval_rounded_predictions = [float(eval_predictions['ood'][i] > 0.5) for i in
                                                range(len(eval_predictions['ood']))]
                id_eval_accuracy = np.mean(np.array(id_eval_rounded_predictions) == np.array(eval_labels['id'].cpu()))
                ood_eval_accuracy = np.mean(
                    np.array(ood_eval_rounded_predictions) == np.array(eval_labels['ood'].cpu()))
                id_eval_roc_auc = roc_auc_score(np.array(eval_labels['id'].cpu()),
                                                np.array(eval_predictions['id'].cpu()))
                ood_eval_roc_auc = roc_auc_score(np.array(eval_labels['ood'].cpu()),
                                                 np.array(eval_predictions['ood'].cpu()))
                run.log({'ID Eval Accuracy': id_eval_accuracy, 'ID Eval ROC-AUC': id_eval_roc_auc,
                         'OOD Eval Accuracy': ood_eval_accuracy, 'OOD Eval ROC-AUC': ood_eval_roc_auc},
                        step=global_step)
            classifier_model.train()
            del eval_losses, eval_predictions, eval_labels, eval_stats
            torch.cuda.empty_cache()
            utils.sync_cuda(sync_cuda_timing)
            eval_wall_sec = time.time() - eval_start_time
            for eval_key in ['id', 'ood']:
                examples_global = accelerator.gather(torch.tensor(
                    [eval_counts[eval_key]['examples']], device=device, dtype=torch.float64)).sum().item()
                tokens_global = accelerator.gather(torch.tensor(
                    [eval_counts[eval_key]['tokens']], device=device, dtype=torch.float64)).sum().item()
                loss_tokens_global = accelerator.gather(torch.tensor(
                    [eval_counts[eval_key]['loss_tokens']], device=device, dtype=torch.float64)).sum().item()
                if accelerator.is_main_process and save_raw_efficiency:
                    utils.append_jsonl(train_eval_metrics_path, {
                        'global_step': int(global_step),
                        'epoch': int(epoch),
                        'eval_split': eval_key,
                        'eval_wall_sec': float(eval_wall_sec),
                        'num_examples': float(examples_global),
                        'num_tokens': float(tokens_global),
                        'num_loss_tokens': float(loss_tokens_global),
                        'examples_per_sec': float(examples_global / max(eval_wall_sec, 1e-12)),
                        'tokens_per_sec': float(tokens_global / max(eval_wall_sec, 1e-12)),
                    })

        if ckpt_freq != -1 and global_step % ckpt_freq == 0:
            classifier_model.eval()
            save_dir = os.path.join(output_dir, 'ckpt_%d' % global_step)
            optimizer_to_save = None
            scheduler_to_save = None
            if save_opt_scheduler:
                optimizer_to_save = optimizer
                scheduler_to_save = scheduler
            utils.save_model(classifier_model, tokenizer, optimizer_to_save, scheduler_to_save, accelerator,
                             save_dir=save_dir, push_to_hub=False)
            classifier_model.train()

if accelerator.is_main_process:
    utils.sync_cuda(sync_cuda_timing)
    total_wall_time = time.time() - start_time
    gpu_hours = total_wall_time / 3600 * accelerator.num_processes
    final_stats = {
        'wall_clock_time_sec': float(total_wall_time),
        'gpu_hours': float(gpu_hours),
        'world_size': int(accelerator.num_processes),
        'num_trainable_params': int(num_trainable_params),
        'num_total_params': int(num_total_params),
        'optimizer_steps': int(optimizer_step_idx),
        'global_step': int(global_step),
        'total_examples': float(total_train_examples),
        'total_tokens': float(total_train_tokens),
        'total_loss_tokens': float(total_train_loss_tokens),
        'total_flops_trainable_est': float(total_train_flops_trainable),
        'total_flops_total_est': float(total_train_flops_total),
        'avg_example_time_sec': float(total_wall_time / max(total_train_examples, 1e-12)),
        'avg_token_time_sec': float(total_wall_time / max(total_train_tokens, 1e-12)),
        'examples_per_sec': float(total_train_examples / max(total_wall_time, 1e-12)),
        'tokens_per_sec': float(total_train_tokens / max(total_wall_time, 1e-12)),
        'peak_memory_allocated_bytes': int(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
    }
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)
