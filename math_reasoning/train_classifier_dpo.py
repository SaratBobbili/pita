import argparse
import json
import os
import time

import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, \
    get_constant_schedule_with_warmup, BitsAndBytesConfig

from peft import get_peft_model, LoraConfig, TaskType

from utils import read_jsonl
from utils_dpo import DpoDataset, dpo_collate_fn, compute_log_probs, dpo_loss, compute_dpo_accuracy

parser = argparse.ArgumentParser(description='DPO Training Script')
parser.add_argument('--ref_model_id', default='meta-llama/Llama-3.2-8B-Instruct', type=str)
parser.add_argument('--train_eval_save_path', required=True, type=str)
parser.add_argument('--data_paths', required=True, nargs='+', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--output_dir', default='checkpoints/temp/', type=str)
parser.add_argument('--num_epochs', default=3, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--warmup_step', default=500, type=int)
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--eta', default=0.1, type=float, help='Beta parameter for DPO loss')
parser.add_argument('--seed', default=47, type=int)
parser.add_argument('--track', default=0, type=int)
parser.add_argument('--wandb_entity', default=None, type=str)
parser.add_argument('--wandb_project', default="", type=str)
parser.add_argument('--wandb_run_name', default="", type=str)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)

args = parser.parse_args()
print(vars(args))

accelerator = Accelerator()
set_seed(args.seed * 42 + accelerator.process_index * 100003)

checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)
resume_from_checkpoint = os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin"))

if accelerator.is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

tokenizer = AutoTokenizer.from_pretrained(args.ref_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    args.ref_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

ref_model = AutoModelForCausalLM.from_pretrained(
    args.ref_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
ref_model.eval()
ref_model.requires_grad_(False)

with open(args.train_eval_save_path, 'r') as f:
    train_eval_problems_d = json.load(f)

problem_position_d = {}
all_data = []
merge_keys = ['fully_guided_predictions', 'fully_guided_predictions_correctness', 'partial_guided_prompts',
              'partial_guided_prompts_tokenized', 'num_response_tokens_in_partial_guided_prompts',
              'partial_guided_responses_tokenized', 'partial_guided_predictions',
              'partial_guided_predictions_correctness']

for data_path in args.data_paths:
    current_data = read_jsonl(data_path)
    for i in range(len(current_data)):
        current_problem = current_data[i]['problem']
        if current_problem not in problem_position_d:
            problem_position_d[current_problem] = len(all_data)
            all_data.append(current_data[i])
        else:
            for k in merge_keys:
                all_data[problem_position_d[current_problem]][k].extend(current_data[i][k])

all_train_data = []
all_eval_data = []
for ex in all_data:
    split = train_eval_problems_d[ex['problem']]['split']
    if split == 'train':
        all_train_data.append(ex)
    elif split == 'eval':
        all_eval_data.append(ex)
    else:
        raise ValueError(f"Unknown split: {split}")

print('Total number of training problems:', len(all_train_data))
print('Total number of eval problems:', len(all_eval_data))

all_train_dpo_data = DpoDataset(all_train_data)
all_eval_dpo_data = DpoDataset(all_eval_data)

print('Total training samples:', len(all_train_dpo_data))
print('Total eval samples:', len(all_eval_dpo_data))

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        return dpo_collate_fn(batch, tokenizer.pad_token_id, tokenizer)
    return collate_fn

collate_fn = create_collate_fn(tokenizer)
train_loader = DataLoader(all_train_dpo_data, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, pin_memory=False)
eval_loader = DataLoader(all_eval_dpo_data, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, pin_memory=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step)

model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, eval_loader, scheduler
)

if resume_from_checkpoint:
    accelerator.print("Loading checkpoint state...")
    accelerator.load_state(checkpoint_dir)
else:
    accelerator.print("No checkpoint found. Starting from scratch.")

run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                 entity=args.wandb_entity, config=vars(args)) if args.track else None

def eval_dpo(model, dataloader, ref_model, beta=0.1):
    model.eval()
    total_accuracy = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, _, _ = dpo_loss(model, batch, beta=beta, ref_model=ref_model)
            total_loss += loss.item()
            accuracy = compute_dpo_accuracy(model, batch, beta=beta, ref_model=ref_model)
            total_accuracy += accuracy
            num_batches += 1

    return total_loss / num_batches, total_accuracy / num_batches

global_step = 0
start_time = time.time()
model.train()

for epoch in range(args.num_epochs):
    print(f"Starting epoch {epoch + 1}/{args.num_epochs}")
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        with accelerator.accumulate(model):
            loss, logp_c, logp_r = dpo_loss(model, batch, beta=args.eta, ref_model=ref_model)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process and run:
                run.log({"dpo_loss": loss.item(), "step": global_step}, step=global_step)

            global_step += 1

    if accelerator.is_main_process:
        eval_loss, eval_accuracy = eval_dpo(model, eval_loader, ref_model, beta=args.eta)
        print(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
        if run:
            run.log({
                'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy,
                'epoch': epoch + 1
            }, step=global_step)

        accelerator.save_state(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")

if accelerator.is_main_process:
    total_wall_time = time.time() - start_time
    gpu_hours = total_wall_time / 3600 * accelerator.num_processes
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens_per_step = args.batch_size * args.max_length
    total_steps = global_step // accelerator.num_processes
    total_flops = 6 * num_params * tokens_per_step * total_steps

    stats = {
        "wall_clock_time_sec": total_wall_time,
        "gpu_hours": gpu_hours,
        "total_flops_est": total_flops,
        "num_params": num_params,
        "tokens_per_step": tokens_per_step,
        "steps": total_steps,
    }

    with open(os.path.join(args.output_dir, "training_stats_.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Training completed in {total_wall_time:.2f} seconds ({gpu_hours:.2f} GPU hours)")
    print(f"Training stats written to {args.output_dir}/training_stats.json")

    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print(f"LoRA adapter saved to {args.output_dir}/final_model")

if run:
    run.finish()
