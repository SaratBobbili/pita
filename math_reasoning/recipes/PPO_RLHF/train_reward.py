"""Pairwise reward model training on 1B Instruct (chosen vs rejected)."""
import json
import os
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

_MR = Path(__file__).resolve().parents[2]
if str(_MR) not in sys.path:
    sys.path.insert(0, str(_MR))
from utils import append_jsonl, count_parameters, ensure_dir


def resolve_root(cfg: DictConfig) -> Path:
    if cfg.math_reasoning_root:
        return Path(cfg.math_reasoning_root)
    return _MR


class EfficiencyCallback(TrainerCallback):
    def __init__(self, efficiency_dir, log_every, num_trainable, num_total, model_id):
        self.efficiency_dir = efficiency_dir
        self.log_every = log_every
        self.num_trainable = num_trainable
        self.num_total = num_total
        self.model_id = model_id
        self.step_path = os.path.join(efficiency_dir, "train_step_metrics.jsonl")
        self.last_step_time = None
        self.optimizer_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_step_time = time.time()
        if state.is_world_process_zero:
            ensure_dir(self.efficiency_dir)
            with open(os.path.join(self.efficiency_dir, "run_metadata.json"), "w") as f:
                json.dump(
                    {
                        "script": "recipes/PPO_RLHF/train_reward.py",
                        "world_size": args.world_size,
                        "num_trainable_params": int(self.num_trainable),
                        "num_total_params": int(self.num_total),
                        "model_id": self.model_id,
                        "dtype": "bfloat16" if args.bf16 else "float32",
                        "start_time_unix": self.last_step_time,
                    },
                    f,
                    indent=2,
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        now = time.time()
        step_wall = now - self.last_step_time
        self.last_step_time = now
        self.optimizer_step += 1
        if self.optimizer_step % self.log_every != 0:
            return
        batch_tokens = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.world_size
        )
        append_jsonl(
            self.step_path,
            {
                "global_step": int(state.global_step),
                "optimizer_step": int(self.optimizer_step),
                "epoch": float(state.epoch) if state.epoch is not None else 0.0,
                "step_wall_sec": float(step_wall),
                "num_examples": float(batch_tokens),
                "num_tokens": float(batch_tokens),
                "num_loss_tokens": float(batch_tokens),
                "flops_trainable_est": float(6.0 * self.num_trainable * batch_tokens),
                "flops_total_est": float(6.0 * self.num_total * batch_tokens),
                "examples_per_sec": float(batch_tokens / max(step_wall, 1e-12)),
                "tokens_per_sec": float(batch_tokens / max(step_wall, 1e-12)),
                "loss_tokens_per_sec": float(batch_tokens / max(step_wall, 1e-12)),
                "learning_rate": float(logs.get("learning_rate", 0.0)),
                "gradient_norm": float(logs.get("grad_norm", 0.0)),
                "train_loss_accumulated": float(logs.get("loss", 0.0)),
            },
        )


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_ids = inputs.pop("chosen_input_ids")
        chosen_mask = inputs.pop("chosen_attention_mask")
        rejected_ids = inputs.pop("rejected_input_ids")
        rejected_mask = inputs.pop("rejected_attention_mask")
        inputs.pop("labels", None)

        chosen_scores = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits.view(-1)
        rejected_scores = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.view(-1)
        loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
        return (loss, (chosen_scores, rejected_scores)) if return_outputs else loss


def tokenize_pair(example, tokenizer, max_length):
    chosen = tokenizer(
        example["prompt"] + example["chosen"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    rejected = tokenizer(
        example["prompt"] + example["rejected"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "labels": 1.0,
    }


@hydra.main(version_base=None, config_path="configs", config_name="train_reward")
def main(cfg: DictConfig):
    root = resolve_root(cfg)
    os.chdir(root)
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    prepared = root / cfg.prepared_data_dir
    if not prepared.exists():
        raise FileNotFoundError(f"Prepared data missing: {prepared}")
    raw = load_from_disk(str(prepared))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds = raw["train"].map(
        lambda ex: tokenize_pair(ex, tokenizer, cfg.max_length),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing train",
        num_proc=4,
    )
    eval_ds = None
    if "eval" in raw:
        eval_ds = raw["eval"].map(
            lambda ex: tokenize_pair(ex, tokenizer, cfg.max_length),
            remove_columns=raw["eval"].column_names,
            desc="Tokenizing eval",
            num_proc=4,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    num_trainable = count_parameters(model, trainable_only=True)
    num_total = count_parameters(model, trainable_only=False)
    output_dir = str(root / cfg.output_dir)
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps,
        report_to=cfg.report_to if cfg.report_to != "none" else [],
        run_name=cfg.run_name,
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    callbacks = []
    if cfg.efficiency.save_raw_efficiency:
        callbacks.append(
            EfficiencyCallback(
                os.path.join(output_dir, "efficiency"),
                int(cfg.efficiency.log_every),
                num_trainable,
                num_total,
                cfg.model_id,
            )
        )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if trainer.accelerator.is_main_process:
        with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
            json.dump(
                {
                    "train_runtime": float(result.metrics.get("train_runtime", 0.0)),
                    "train_loss": float(result.metrics.get("train_loss", 0.0)),
                    "num_trainable_params": int(num_trainable),
                    "num_total_params": int(num_total),
                },
                f,
                indent=2,
            )
        print(f"Saved reward model to {output_dir}")


if __name__ == "__main__":
    main()
