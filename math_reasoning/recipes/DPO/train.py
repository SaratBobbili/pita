"""Full-parameter DPO training with TRL DPOTrainer + DeepSpeed ZeRO-3."""
import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, set_seed
from trl import DPOConfig, DPOTrainer

_MR = Path(__file__).resolve().parents[2]
if str(_MR) not in sys.path:
    sys.path.insert(0, str(_MR))
from utils import append_jsonl, ensure_dir


def resolve_root(cfg: DictConfig) -> Path:
    if cfg.math_reasoning_root:
        return Path(cfg.math_reasoning_root)
    return _MR


def count_zero3_parameters(model, trainable_only=False):
    return sum(
        getattr(p, "ds_numel", p.numel())
        for p in model.parameters()
        if not trainable_only or p.requires_grad
    )


class StableCacheDPOTrainer(DPOTrainer):
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches, device)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def log(self, logs, start_time=None):
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return Trainer.log(self, logs, start_time)

    def __getstate__(self):
        keys = (
            "is_encoder_decoder",
            "is_vision_model",
            "processor",
            "processing_class",
            "max_length",
            "max_prompt_length",
            "max_target_length",
            "truncation_mode",
            "label_pad_token_id",
        )
        return {key: getattr(self, key, None) for key in keys}

    def __setstate__(self, state):
        self.__dict__.update(state)


class EfficiencyCallback(TrainerCallback):
    def __init__(self, efficiency_dir, log_every, num_trainable, num_total, model_id):
        self.efficiency_dir = efficiency_dir
        self.log_every = log_every
        self.num_trainable = num_trainable
        self.num_total = num_total
        self.model_id = model_id
        self.step_path = os.path.join(efficiency_dir, "train_step_metrics.jsonl")
        self.start_time = None
        self.last_step_time = None
        self.optimizer_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_step_time = self.start_time
        if state.is_world_process_zero:
            ensure_dir(self.efficiency_dir)
            with open(os.path.join(self.efficiency_dir, "run_metadata.json"), "w") as f:
                json.dump(
                    {
                        "script": "recipes/DPO/train.py",
                        "world_size": args.world_size,
                        "num_trainable_params": int(self.num_trainable),
                        "num_total_params": int(self.num_total),
                        "model_id": self.model_id,
                        "dtype": "bfloat16" if args.bf16 else "float32",
                        "start_time_unix": self.start_time,
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
        batch_tokens = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
        append_jsonl(
            self.step_path,
            {
                "global_step": int(state.global_step),
                "optimizer_step": int(self.optimizer_step),
                "epoch": float(state.epoch) if state.epoch is not None else 0.0,
                "step_wall_sec": float(step_wall),
                "num_examples": float(batch_tokens),
                "num_tokens": float(batch_tokens * args.max_length if hasattr(args, "max_length") else batch_tokens),
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


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    root = resolve_root(cfg)
    os.chdir(root)
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    os.environ["WANDB_PROJECT"] = cfg.wandb_project

    prepared = root / cfg.prepared_data_dir
    if not prepared.exists():
        raise FileNotFoundError(
            f"Prepared data not found at {prepared}. Run prepare_data.py first."
        )
    dataset = load_from_disk(str(prepared))
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"] if "eval" in dataset else None

    output_dir = str(root / cfg.output_dir)
    deepspeed_config = str(root / cfg.deepspeed_config)
    ensure_dir(output_dir)

    # DPOConfig must exist before from_pretrained: TrainingArguments installs
    # HfDeepSpeedConfig, which makes model construction happen inside zero.Init.
    training_args = DPOConfig(
        output_dir=output_dir,
        deepspeed=deepspeed_config,
        beta=cfg.beta,
        loss_type="sigmoid",
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy if eval_dataset is not None else "no",
        eval_steps=cfg.eval_steps,
        report_to=cfg.report_to if cfg.report_to != "none" else [],
        run_name=cfg.run_name,
        seed=cfg.seed,
        remove_unused_columns=False,
        precompute_ref_log_probs=cfg.precompute_ref_log_probs,
        dataset_num_proc=cfg.dataset_num_proc,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float32,
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.ref_model_id, **model_kwargs)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    num_trainable = count_zero3_parameters(model, trainable_only=True)
    num_total = count_zero3_parameters(model, trainable_only=False)
    print(f"trainable params: {num_trainable:,}  total: {num_total:,}")

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    callbacks = []
    if cfg.efficiency.save_raw_efficiency:
        callbacks.append(
            EfficiencyCallback(
                efficiency_dir=os.path.join(output_dir, "efficiency"),
                log_every=int(cfg.efficiency.log_every),
                num_trainable=num_trainable,
                num_total=num_total,
                model_id=cfg.model_id,
            )
        )

    trainer = StableCacheDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if trainer.accelerator.is_main_process:
        wall = float(train_result.metrics.get("train_runtime", 0.0))
        world_size = int(trainer.args.world_size)
        stats = {
            "train_runtime": wall,
            "wall_clock_time_sec": wall,
            "gpu_hours": wall / 3600.0 * world_size,
            "world_size": world_size,
            "train_samples_per_second": float(
                train_result.metrics.get("train_samples_per_second", 0.0)
            ),
            "train_steps_per_second": float(
                train_result.metrics.get("train_steps_per_second", 0.0)
            ),
            "examples_per_sec": float(
                train_result.metrics.get("train_samples_per_second", 0.0)
            ),
            "train_loss": float(train_result.metrics.get("train_loss", 0.0)),
            "num_trainable_params": int(num_trainable),
            "num_total_params": int(num_total),
        }
        with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved model and stats to {output_dir}")


if __name__ == "__main__":
    main()
