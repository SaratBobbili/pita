"""PPO/RLHF: 8B policy, frozen 1B reward, trainable 1B value (Accelerate MULTI_GPU)."""
import json
import os
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    set_seed,
)

_MR = Path(__file__).resolve().parents[2]
if str(_MR) not in sys.path:
    sys.path.insert(0, str(_MR))
from utils import append_jsonl, count_parameters, ensure_dir, get_message


def resolve_root(cfg: DictConfig) -> Path:
    if cfg.math_reasoning_root:
        return Path(cfg.math_reasoning_root)
    return _MR


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def format_prompt(tokenizer, prompt, use_chat_template):
    if use_chat_template:
        return tokenizer.apply_chat_template(
            get_message(prompt), add_generation_prompt=True, tokenize=False
        )
    return prompt


@torch.no_grad()
def score_sequences(reward_model, tokenizer, texts, max_length, device):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = reward_model(**enc).logits.view(-1)
    return logits


def response_logprobs(model, input_ids, attention_mask, prompt_lens):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].clone().float()
    for i, plen in enumerate(prompt_lens):
        # tokens before response end at prompt_len-1 in labels index space
        cut = max(int(plen) - 1, 0)
        mask[i, :cut] = 0
    seq_logp = (token_logp * mask).sum(dim=1)
    return seq_logp, mask


def ppo_policy_loss(logp, old_logp, advantages, cliprange):
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    return -torch.min(unclipped, clipped).mean()


def ppo_value_loss(values, old_values, returns, cliprange_value):
    v_clipped = old_values + torch.clamp(
        values - old_values, -cliprange_value, cliprange_value
    )
    loss_unclipped = (values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()


@hydra.main(version_base=None, config_path="configs", config_name="train_ppo")
def main(cfg: DictConfig):
    root = resolve_root(cfg)
    os.chdir(root)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.bf16 else "no",
    )
    set_seed(cfg.seed + accelerator.process_index)
    if accelerator.is_main_process:
        print(OmegaConf.to_yaml(cfg))

    prepared = root / cfg.prepared_data_dir
    raw = load_from_disk(str(prepared))
    prompts = list(dict.fromkeys(raw["train"]["prompt"]))
    if accelerator.is_main_process:
        print(f"Unique train prompts: {len(prompts)}")

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(cfg.policy_model_id, torch_dtype=dtype)
    ref = AutoModelForCausalLM.from_pretrained(cfg.ref_model_id, torch_dtype=dtype)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model_path, num_labels=1, torch_dtype=dtype
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    value_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.value_model_id, num_labels=1, torch_dtype=dtype
    )
    if cfg.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    output_dir = str(root / cfg.output_dir)
    if accelerator.is_main_process:
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, "efficiency"))
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        with open(os.path.join(output_dir, "efficiency", "run_metadata.json"), "w") as f:
            json.dump(
                {
                    "script": "recipes/PPO_RLHF/train_ppo.py",
                    "world_size": accelerator.num_processes,
                    "num_trainable_params": int(count_parameters(policy, True) + count_parameters(value_model, True)),
                    "num_total_params": int(count_parameters(policy, False)),
                    "policy_model_id": cfg.policy_model_id,
                    "reward_model_path": cfg.reward_model_path,
                    "value_model_id": cfg.value_model_id,
                    "dtype": "bfloat16" if cfg.bf16 else "float32",
                    "start_time_unix": time.time(),
                },
                f,
                indent=2,
            )

    prompt_loader = DataLoader(
        PromptDataset(prompts),
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        drop_last=True,
    )

    policy_opt = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    value_opt = torch.optim.AdamW(value_model.parameters(), lr=cfg.value_learning_rate)
    policy_sched = get_constant_schedule_with_warmup(policy_opt, num_warmup_steps=20)
    value_sched = get_constant_schedule_with_warmup(value_opt, num_warmup_steps=20)

    (
        policy,
        value_model,
        ref,
        reward_model,
        policy_opt,
        value_opt,
        prompt_loader,
        policy_sched,
        value_sched,
    ) = accelerator.prepare(
        policy,
        value_model,
        ref,
        reward_model,
        policy_opt,
        value_opt,
        prompt_loader,
        policy_sched,
        value_sched,
    )

    step_metrics_path = os.path.join(output_dir, "efficiency", "train_step_metrics.jsonl")
    prompt_iter = iter(prompt_loader)
    global_step = 0
    start_time = time.time()

    bar = tqdm(range(cfg.num_iterations), disable=not accelerator.is_local_main_process)
    for iteration in bar:
        try:
            batch_prompts = next(prompt_iter)
        except StopIteration:
            prompt_iter = iter(prompt_loader)
            batch_prompts = next(prompt_iter)

        step_t0 = time.time()
        formatted = [
            format_prompt(tokenizer, p, cfg.use_chat_template) for p in batch_prompts
        ]
        prompt_enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        prompt_enc = {k: v.to(accelerator.device) for k, v in prompt_enc.items()}
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()

        unwrapped_policy = accelerator.unwrap_model(policy)
        with torch.no_grad():
            gen = unwrapped_policy.generate(
                **prompt_enc,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        # rebuild left-padded batch for logprob/value/reward
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        full_enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length + cfg.max_new_tokens,
        )
        full_enc = {k: v.to(accelerator.device) for k, v in full_enc.items()}
        # approximate prompt lengths in re-tokenized sequences
        re_prompt = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        re_prompt_lens = re_prompt["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            old_logp, _ = response_logprobs(
                policy, full_enc["input_ids"], full_enc["attention_mask"], re_prompt_lens
            )
            ref_logp, _ = response_logprobs(
                ref, full_enc["input_ids"], full_enc["attention_mask"], re_prompt_lens
            )
            rewards = score_sequences(
                accelerator.unwrap_model(reward_model),
                tokenizer,
                texts,
                cfg.max_prompt_length + cfg.max_new_tokens,
                accelerator.device,
            )
            values = (
                accelerator.unwrap_model(value_model)(
                    input_ids=full_enc["input_ids"],
                    attention_mask=full_enc["attention_mask"],
                )
                .logits.view(-1)
            )
            kl = old_logp - ref_logp
            scores = rewards - cfg.kl_coef * kl
            advantages = scores - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = scores
            old_values = values.clone()

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        for _ in range(cfg.num_ppo_epochs):
            with accelerator.accumulate(policy), accelerator.accumulate(value_model):
                logp, _ = response_logprobs(
                    policy,
                    full_enc["input_ids"],
                    full_enc["attention_mask"],
                    re_prompt_lens,
                )
                v_pred = value_model(
                    input_ids=full_enc["input_ids"],
                    attention_mask=full_enc["attention_mask"],
                ).logits.view(-1)

                p_loss = ppo_policy_loss(
                    logp, old_logp.detach(), advantages.detach(), cfg.cliprange
                )
                v_loss = ppo_value_loss(
                    v_pred, old_values.detach(), returns.detach(), cfg.cliprange_value
                )
                loss = p_loss + cfg.vf_coef * v_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    accelerator.clip_grad_norm_(value_model.parameters(), cfg.max_grad_norm)
                policy_opt.step()
                value_opt.step()
                policy_sched.step()
                value_sched.step()
                policy_opt.zero_grad(set_to_none=True)
                value_opt.zero_grad(set_to_none=True)
                policy_loss_acc += float(p_loss.detach())
                value_loss_acc += float(v_loss.detach())

        global_step += 1
        step_wall = time.time() - step_t0
        mean_reward = float(accelerator.gather(rewards.detach()).mean())
        mean_kl = float(accelerator.gather(kl.detach()).mean())

        if accelerator.is_main_process:
            bar.set_postfix(reward=f"{mean_reward:.3f}", kl=f"{mean_kl:.3f}")
            if cfg.efficiency.save_raw_efficiency and global_step % int(cfg.efficiency.log_every) == 0:
                n = cfg.per_device_batch_size * accelerator.num_processes
                append_jsonl(
                    step_metrics_path,
                    {
                        "global_step": int(global_step),
                        "optimizer_step": int(global_step),
                        "epoch": float(iteration / max(cfg.num_iterations, 1)),
                        "step_wall_sec": float(step_wall),
                        "num_examples": float(n),
                        "num_tokens": float(n * cfg.max_new_tokens),
                        "num_loss_tokens": float(n * cfg.max_new_tokens),
                        "flops_trainable_est": 0.0,
                        "flops_total_est": 0.0,
                        "examples_per_sec": float(n / max(step_wall, 1e-12)),
                        "tokens_per_sec": float(n * cfg.max_new_tokens / max(step_wall, 1e-12)),
                        "loss_tokens_per_sec": float(n * cfg.max_new_tokens / max(step_wall, 1e-12)),
                        "learning_rate": float(policy_sched.get_last_lr()[0]),
                        "gradient_norm": 0.0,
                        "train_loss_accumulated": float(policy_loss_acc / cfg.num_ppo_epochs),
                        "reward_mean": mean_reward,
                        "kl_mean": mean_kl,
                        "value_loss": float(value_loss_acc / cfg.num_ppo_epochs),
                    },
                )

        if accelerator.is_main_process and global_step % int(cfg.save_every) == 0:
            save_dir = os.path.join(output_dir, f"ckpt_{global_step}")
            ensure_dir(save_dir)
            accelerator.unwrap_model(policy).save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            accelerator.unwrap_model(value_model).save_pretrained(
                os.path.join(save_dir, "value_model")
            )

    if accelerator.is_main_process:
        accelerator.unwrap_model(policy).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        accelerator.unwrap_model(value_model).save_pretrained(
            os.path.join(output_dir, "value_model")
        )
        with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
            json.dump(
                {
                    "wall_clock_time_sec": time.time() - start_time,
                    "num_iterations": int(cfg.num_iterations),
                    "global_step": int(global_step),
                },
                f,
                indent=2,
            )
        print(f"Saved PPO policy to {output_dir}")


if __name__ == "__main__":
    main()
