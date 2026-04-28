import argparse
import json
import os
import socket
import time

import numpy as np
import math

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, DataCollatorForLanguageModeling, \
    get_constant_schedule_with_warmup

import utils
from classifier import CustomLlamaForSequenceClassification
from utils import read_jsonl, custom_collate_fn as base_custom_collate_fn
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
parser.add_argument('--resume_opt_scheduler', default=0, type=int,
                    help='whether to resume optimizer and scheduler from the checkpoint, 0 (no), 1 (yes)')
parser.add_argument('--train_eval_save_path', required=False, type=str, help='')
parser.add_argument('--init_mode', required=True, type=str,
                    help='zero / random / reuse / warmstart init the output layer for the classifier. For second round or higher, warmstart is reusing the previous ckpt without modifying any weight.')
parser.add_argument('--inference_mode', required=True, type=str,
                    help='inference mode supported by the classifier. First round does not matter')
parser.add_argument('--loss_type', default='bce', type=str, help='loss type for the classifier, bce or mse')
parser.add_argument('--use_bias', default=0, type=int,
                    help='whether to use bias for the classification layer, llama 3 does not have bias')
parser.add_argument('--dataset_type', required=True, type=str, help='alpaca_eval or hh_rlhf')
parser.add_argument('--data_path', required=True, type=str, help='path to the training data')
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
parser.add_argument('--lr', default=5e-6, type=float, help='learning rate for the classifier')
parser.add_argument('--warmup_step', default=-1, type=int, help='warmup steps for the classifier, -1 means no warmup')
parser.add_argument('--weight_decay', default=5e-2, type=float, help='weight decay for the classifier')
parser.add_argument('--eval_freq', default=500, type=int, help='evaluation frequency')
parser.add_argument('--ckpt_freq', default=5000, type=int, help='checkpoint frequency')
parser.add_argument('--save_opt_scheduler', default=0, type=int, help='whether to save optimizer and scheduler state')
parser.add_argument('--seed', default=47, type=int, help='seed for reproduction')
parser.add_argument('--pairwise_margin_reg', default=1e-3, type=float,
                    help='L2 regularization weight on pairwise margin to reduce overconfident overfitting')
parser.add_argument('--early_stop_patience', default=-1, type=int,
                    help='number of eval rounds without ID eval loss improvement before stopping; -1 disables early stopping')
parser.add_argument('--early_stop_min_delta', default=1e-4, type=float,
                    help='minimum ID eval loss improvement required to reset early stopping patience')
parser.add_argument('--track', default=0, type=int, help='whether to report to wandb')
parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity')
parser.add_argument('--wandb_project', default="", type=str, help='wandb project')
parser.add_argument('--wandb_run_name', default="", type=str, help='wandb run name')
parser.add_argument('--num_atoms', default=11, type=int, help='number of atoms for mle classifier')
parser.add_argument('--V_min', default=0, type=float, help='V_min for histogram learning')
parser.add_argument('--V_max', default=1, type=float, help='V_max for histogram learning')
parser.add_argument('--max_length', default=-1, type=int, help='max tokens for training')
parser.add_argument('--num_mlp_layers', default=0, type=int,
                    help='number of Linear+GELU blocks in the classifier MLP residual branch; 0 means plain linear head')
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    help='learning rate schedule after warmup: constant or cosine')
parser.add_argument('--warmup_ratio', default=0.1, type=float,
                    help='warmup ratio used when warmup_step is -1; warmup_step takes precedence when non-negative')
parser.add_argument('--min_lr_ratio', default=0.2, type=float,
                    help='minimum LR ratio after cosine decay; keeps a non-zero floor for gentler late training')
parser.add_argument('--allow_train_holdout_eval_fallback', default=1, type=int,
                    help='fallback to train holdout eval if true eval split has no pairwise rows in data_path')

args = parser.parse_args()
print(socket.gethostname())
# print(vars(args)) # TODO: Re-enable this print statement

world_size = args.world_size
ref_model_id = args.ref_model_id
classifier_model_id = args.classifier_model_id
classifier_ckpt_path = args.classifier_ckpt_path
resume_opt_scheduler = args.resume_opt_scheduler
train_eval_save_path = args.train_eval_save_path # TODO: Figure out what this is for and whether I need it
init_mode = args.init_mode # NOTE: This is irrelevant because we are using V-type
inference_mode = args.inference_mode
loss_type = args.loss_type
use_bias = bool(args.use_bias)
dataset_type = args.dataset_type
data_path = args.data_path # Points to alpaca_noisy_multi_preference_train_eval.json or anthropic_hh_train_eval.json
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
drop_no_variation = bool(args.drop_no_variation) # NOTE: This is unused
id_eval_ratio = args.id_eval_ratio
eta = args.eta # NOTE: Unused
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
pairwise_margin_reg = args.pairwise_margin_reg
early_stop_patience = args.early_stop_patience
early_stop_min_delta = args.early_stop_min_delta

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

# TODO: Finish replacing old values
if dataset_type == 'alpaca_eval':
    dataset_type = 'AE'
    reward_key = 'partial_guided_predictions_correctness'
elif dataset_type == 'hh_rlhf':
    dataset_type = 'HH'
    reward_key = 'partial_guided_predictions_correctness'
else:
    raise ValueError('Unknown dataset name: %s' % dataset_type)

with open(train_eval_save_path, 'r') as f:
    train_eval_problems_d = json.load(f)

# The keys for train_eval_problems_d are the prompts themselves; we want a list of dicts which 
# matches the .jsonl format which the data collected from collect_training_data.py is in
all_data = [{"problem": k} | v for k, v in train_eval_problems_d.items()]

# group all data_paths data
problem_position_d = {}
all_data = []
merge_keys = ['fully_guided_predictions', 'fully_guided_predictions_correctness', 'partial_guided_prompts',
              'partial_guided_prompts_tokenized', 'num_response_tokens_in_partial_guided_prompts',
              'partial_guided_responses_tokenized', 'partial_guided_predictions',
              'partial_guided_predictions_correctness', 'fully_guided_responses_tokenized']
data_path = args.data_path
current_data = read_jsonl(data_path)
for i in range(len(current_data)):
    current_problem = current_data[i]['problem']
    if current_problem not in problem_position_d:
        problem_position_d[current_problem] = len(all_data)
        all_data.append(current_data[i])
    else:
        for k in merge_keys:
            all_data[problem_position_d[current_problem]][k].extend(current_data[i][k])

# Build BCE labels directly from pair preference for the partial-guided candidate.
# preference == 2 means output_2 (mapped to partial-guided) is preferred.
for i in range(len(all_data)):
    current_data = all_data[i]
    assert 'reward' not in current_data
    preference = int(current_data['preference'])
    assert preference in [1, 2], f"Unexpected preference value: {preference}"
    current_data['reward'] = [1 if preference == 2 else 0]

if loss_type == "bce":
    all_rewards = list(x['reward'] for x in all_data)
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

def build_pairwise_data(data_split, use_all_ref_tokens, max_length):
    pairwise_data = []
    for sample in tqdm(data_split, desc='build_pairwise_data'):
        prompt_tokenized = sample['partial_guided_prompts_tokenized'][0]
        full_response_tokenized = sample['fully_guided_responses_tokenized'][0]
        partial_response_tokenized = sample['partial_guided_responses_tokenized'][0]
        preference = int(sample['preference'])
        assert preference in [1, 2], f"Unexpected preference value: {preference}"
        chosen_response = full_response_tokenized if preference == 1 else partial_response_tokenized
        rejected_response = partial_response_tokenized if preference == 1 else full_response_tokenized

        def build_one_side(response_tokenized):
            input_ids = prompt_tokenized[:-1]
            if use_all_ref_tokens == 0:
                target_ids = [prompt_tokenized[-1]]
            else:
                target_ids = [prompt_tokenized[-1]] + response_tokenized
            if max_length != -1:
                if len(input_ids) >= max_length - 1:
                    return None
                if len(input_ids) + len(target_ids) > max_length:
                    target_ids = target_ids[:max_length - len(input_ids)]
            if len(target_ids) == 0:
                return None
            return {'input_ids': input_ids, 'target_ids': target_ids}

        chosen_side = build_one_side(chosen_response)
        rejected_side = build_one_side(rejected_response)
        if chosen_side is None or rejected_side is None:
            continue
        pairwise_data.append({
            'chosen_input_ids': chosen_side['input_ids'],
            'chosen_target_ids': chosen_side['target_ids'],
            'rejected_input_ids': rejected_side['input_ids'],
            'rejected_target_ids': rejected_side['target_ids'],
        })
    return pairwise_data


class PairwiseClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pairwise_collate_fn(batch, pad_token_id):
    def build_single_side(side):
        single_side_batch = [{
            'input_ids': x[f'{side}_input_ids'],
            'target_ids': x[f'{side}_target_ids'],
            'rewards': 0.0,
            'loss_weights': 1.0,
        } for x in batch]
        return base_custom_collate_fn(single_side_batch, pad_token_id)

    return {
        'chosen': build_single_side('chosen'),
        'rejected': build_single_side('rejected'),
    }


pairwise_train_data = build_pairwise_data(all_train_data, use_all_ref_tokens, args.max_length)
train_pairwise_data = pairwise_train_data
eval_pairwise_data = build_pairwise_data(all_eval_data, use_all_ref_tokens, args.max_length)
eval_source = 'true_eval_split'

if len(eval_pairwise_data) == 0 and bool(args.allow_train_holdout_eval_fallback):
    pairwise_train_length = len(pairwise_train_data)
    shuffled_indices = np.random.choice(pairwise_train_length, pairwise_train_length, replace=False)
    holdout_start = int(pairwise_train_length * (1 - id_eval_ratio))
    train_indices = shuffled_indices[:holdout_start]
    eval_indices = shuffled_indices[holdout_start:]
    train_pairwise_data = [pairwise_train_data[i] for i in train_indices]
    eval_pairwise_data = [pairwise_train_data[i] for i in eval_indices]
    eval_source = 'train_holdout_fallback'
    print('true eval split has no pairwise rows in data_path; using train holdout fallback for eval')

if eval_max_size != -1 and eval_max_size < len(eval_pairwise_data):
    eval_random_indices = np.random.choice(len(eval_pairwise_data), eval_max_size, replace=False)
    eval_pairwise_data = [eval_pairwise_data[i] for i in eval_random_indices]
    print('eval indices sum', np.sum(eval_random_indices))

print('total number of training samples', len(train_pairwise_data))
print('total number of eval samples', len(eval_pairwise_data))
print('eval source', eval_source)
assert len(train_pairwise_data) > 0, 'No pairwise training samples after preprocessing.'
if eval_freq != -1:
    assert len(eval_pairwise_data) > 0, 'No eval pairwise samples after preprocessing.'

train_classifier_dataset = PairwiseClassifierDataset(train_pairwise_data)
eval_classifier_dataset = PairwiseClassifierDataset(eval_pairwise_data)
collate_fn = partial(pairwise_collate_fn, pad_token_id=tokenizer.pad_token_id)
assert max_batch_num_tokens == -1, 'pairwise ranking currently supports max_batch_num_tokens == -1 only'
train_classifier_loader = DataLoader(
    train_classifier_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
)
eval_classifier_loader = DataLoader(
    eval_classifier_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
)
print("Finished creating pairwise dataloader")

model_loading_kwargs = {}
if dtype == 'bfloat16':
    model_loading_kwargs['torch_dtype'] = torch.bfloat16
classifier_model = CustomLlamaForSequenceClassification.from_pretrained(classifier_ckpt_path, **model_loading_kwargs,
                                                                        num_labels=vocab_size,
                                                                        loss_type=loss_type, use_bias=use_bias,
                                                                        classifier_type=args.classifier_type,
                                                                        device_map=device, num_atoms=args.num_atoms,
                                                                        V_min=args.V_min, V_max=args.V_max,
                                                                        num_mlp_layers=args.num_mlp_layers)
print("Loaded classifier model")

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
total_training_steps = num_epochs * len(train_classifier_loader) // (gradient_accumulation_step * world_size)
assert total_training_steps > 0, 'total_training_steps must be positive'
if warmup_step == -1:
    computed_warmup_step = max(1, int(total_training_steps * args.warmup_ratio))
else:
    computed_warmup_step = warmup_step
computed_warmup_step = min(computed_warmup_step, max(total_training_steps - 1, 0))
if args.lr_schedule == 'cosine':
    def cosine_with_floor_lr_lambda(current_step):
        if computed_warmup_step > 0 and current_step < computed_warmup_step:
            return float(current_step + 1) / float(computed_warmup_step)
        progress = float(current_step - computed_warmup_step) / float(max(1, total_training_steps - computed_warmup_step))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
    scheduler = LambdaLR(optimizer, cosine_with_floor_lr_lambda)
else:
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=computed_warmup_step)
print(f'LR schedule: {args.lr_schedule}, total_training_steps: {total_training_steps}, warmup_steps: {computed_warmup_step}, min_lr_ratio: {args.min_lr_ratio}')
if resume_opt_scheduler == 1:
    optimizer_scheduler_state = torch.load(os.path.join(classifier_ckpt_path, 'optimizer_scheduler_state.pt'))
    optimizer.load_state_dict(optimizer_scheduler_state['optimizer'])
    scheduler.load_state_dict(optimizer_scheduler_state['scheduler'])
    print('optimizer and scheduler resumed from the checkpoint')

global_step = 0
start_time = time.time()
accumulated_loss = torch.tensor(0.0).to(device)
accumulated_pairwise_acc = torch.tensor(0.0).to(device)
accumulated_margin = torch.tensor(0.0).to(device)
accumulated_margin_penalty = torch.tensor(0.0).to(device)
accumulated_train_batches = 0
best_eval_loss = float('inf')
best_step = -1
bad_eval_rounds = 0
stop_training = False

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

classifier_model, optimizer, train_classifier_loader, eval_classifier_loader, scheduler = \
    accelerator.prepare(classifier_model, optimizer, train_classifier_loader, eval_classifier_loader,
                        scheduler)


def get_sequence_scores(logits, loss_mask):
    logits = logits.squeeze(-1)
    loss_mask = loss_mask.float()
    return (logits * loss_mask).sum(dim=1) / torch.clamp(loss_mask.sum(dim=1), min=1.0)


for epoch in range(num_epochs):
    run.log({'Epoch': epoch}, step=global_step)
    bar = tqdm(train_classifier_loader) if accelerator.is_local_main_process else train_classifier_loader
    torch.cuda.empty_cache()
    for batch_input_data in bar:
        global_step += world_size
        chosen_batch = batch_input_data['chosen']
        rejected_batch = batch_input_data['rejected']
        chosen_outputs = classifier_model(
            input_ids=chosen_batch['input_ids'],
            attention_mask=chosen_batch['attention_mask'],
            labels=chosen_batch['rewards'],
            loss_mask=chosen_batch['loss_mask'],
            loss_weights=chosen_batch['loss_weights'],
        )
        rejected_outputs = classifier_model(
            input_ids=rejected_batch['input_ids'],
            attention_mask=rejected_batch['attention_mask'],
            labels=rejected_batch['rewards'],
            loss_mask=rejected_batch['loss_mask'],
            loss_weights=rejected_batch['loss_weights'],
        )
        chosen_scores = get_sequence_scores(chosen_outputs.logits, chosen_batch['loss_mask'])
        rejected_scores = get_sequence_scores(rejected_outputs.logits, rejected_batch['loss_mask'])
        margins = chosen_scores - rejected_scores
        pairwise_loss = -F.logsigmoid(margins).mean()
        margin_penalty = pairwise_margin_reg * torch.square(margins).mean()
        objective = pairwise_loss + margin_penalty
        loss = objective / gradient_accumulation_step  # normalize for grad accumulation
        del batch_input_data, chosen_batch, rejected_batch, chosen_outputs, rejected_outputs
        torch.cuda.empty_cache()

        accelerator.backward(loss)
        accumulated_loss += loss.detach()
        accumulated_pairwise_acc += (margins > 0).float().mean().detach()
        accumulated_margin += margins.mean().detach()
        accumulated_margin_penalty += margin_penalty.detach()
        accumulated_train_batches += 1

        # Logging purposes
        if (global_step // world_size) % gradient_accumulation_step == 0:
            if args.classifier_type == "Q":
                grad_norm = torch.tensor(0.0).to(device)
            else:
                grad_norm = accelerator.clip_grad_norm_(classifier_model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            elapsed_time = time.time() - start_time
            run.log({
                'Training Loss': accelerator.gather(accumulated_loss).mean(),
                'Learning Rate': scheduler.get_last_lr()[0],
                'Steps per Min': global_step / (elapsed_time / 60),
                'Gradient Norm': accelerator.gather(grad_norm).mean(),
                'Train Pairwise Accuracy': accelerator.gather(
                    accumulated_pairwise_acc / max(accumulated_train_batches, 1)).mean(),
                'Train Margin Mean': accelerator.gather(
                    accumulated_margin / max(accumulated_train_batches, 1)).mean(),
                'Train Margin Penalty': accelerator.gather(
                    accumulated_margin_penalty / max(accumulated_train_batches, 1)).mean(),
            }, step=global_step)
            accumulated_loss = torch.tensor(0.0).to(device)
            accumulated_pairwise_acc = torch.tensor(0.0).to(device)
            accumulated_margin = torch.tensor(0.0).to(device)
            accumulated_margin_penalty = torch.tensor(0.0).to(device)
            accumulated_train_batches = 0

        if eval_freq != -1 and (global_step % eval_freq == 0 or global_step == 1):
            classifier_model.eval()
            eval_losses = {'eval': []}
            eval_accuracies = {'eval': []}
            eval_margins = {'eval': []}
            with torch.no_grad():
                for eval_key, eval_loader in [('eval', eval_classifier_loader)]:
                    for batch_input_data in eval_loader:
                        chosen_batch = batch_input_data['chosen']
                        rejected_batch = batch_input_data['rejected']
                        chosen_outputs = classifier_model(
                            input_ids=chosen_batch['input_ids'],
                            attention_mask=chosen_batch['attention_mask'],
                            labels=chosen_batch['rewards'],
                            loss_mask=chosen_batch['loss_mask'],
                            loss_weights=chosen_batch['loss_weights'],
                        )
                        rejected_outputs = classifier_model(
                            input_ids=rejected_batch['input_ids'],
                            attention_mask=rejected_batch['attention_mask'],
                            labels=rejected_batch['rewards'],
                            loss_mask=rejected_batch['loss_mask'],
                            loss_weights=rejected_batch['loss_weights'],
                        )
                        chosen_scores = get_sequence_scores(chosen_outputs.logits, chosen_batch['loss_mask'])
                        rejected_scores = get_sequence_scores(rejected_outputs.logits, rejected_batch['loss_mask'])
                        margins = chosen_scores - rejected_scores
                        loss = -F.logsigmoid(margins).mean()
                        accuracy = (margins > 0).float().mean()
                        eval_losses[eval_key].append(loss)
                        eval_accuracies[eval_key].append(accuracy)
                        eval_margins[eval_key].append(margins.mean())

                        del batch_input_data, chosen_batch, rejected_batch, chosen_outputs, rejected_outputs
                        torch.cuda.empty_cache()

            eval_losses = {k: torch.mean(accelerator.gather(torch.stack(v))) for k, v in eval_losses.items()}
            eval_accuracies = {k: torch.mean(accelerator.gather(torch.stack(v))) for k, v in eval_accuracies.items()}
            eval_margins = {k: torch.mean(accelerator.gather(torch.stack(v))) for k, v in eval_margins.items()}
            run.log({
                'Eval Pairwise Loss': eval_losses['eval'],
                'Eval Pairwise Accuracy': eval_accuracies['eval'],
                'Eval Margin Mean': eval_margins['eval'],
            }, step=global_step)
            current_eval_loss = eval_losses['eval'].item()
            if current_eval_loss < best_eval_loss - early_stop_min_delta:
                best_eval_loss = current_eval_loss
                best_step = global_step
                bad_eval_rounds = 0
            else:
                bad_eval_rounds += 1
            run.log({
                'Best Eval Pairwise Loss': best_eval_loss,
                'Best Step': best_step,
                'Early Stop Bad Rounds': bad_eval_rounds,
            }, step=global_step)
            if early_stop_patience != -1 and bad_eval_rounds >= early_stop_patience:
                print(f'Early stopping at step {global_step}: best_eval_loss={best_eval_loss:.6f} (step {best_step})')
                stop_training = True
            classifier_model.train()
            del eval_losses, eval_accuracies, eval_margins
            torch.cuda.empty_cache()

        if stop_training:
            break

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

    if stop_training:
        break
