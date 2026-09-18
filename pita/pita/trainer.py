"""Accelerate training loop for the value classifier.

Derived from ``refactor_old/training/trainer.py``. Changes: Hydra config objects replaced
by the dataclasses in :mod:`pita.configs`, the per-architecture ``get_classifier_class``
registry replaced by :class:`pita.classifier.ValueClassifier`, and DeepSpeed dropped --
the trainable model is at most ~2B parameters, so plain DDP across 8 GPUs is enough and
removes a whole configuration surface.

The reference policy is never loaded here. Only the classifier trains; the policy is
frozen by construction, which is the entire point of inference-time alignment.
"""

import json
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pita.classifier import ValueClassifier, validate_pair
from pita.data import (
    ClassifierDataset,
    build_examples,
    collate,
    explained_variance,
    r_squared,
    read_pairs,
    roc_auc,
    split_by_prompt,
)


def build_classifier(model_args, ref_vocab_size):
    """Round 1 starts from the backbone; later rounds resume the previous checkpoint."""
    dtype = getattr(torch, model_args.dtype)
    if model_args.classifier_path:
        return ValueClassifier.from_pretrained(model_args.classifier_path, dtype=dtype)

    classifier = ValueClassifier.from_backbone(
        model_args.classifier_model_id,
        ref_vocab_size=ref_vocab_size,
        dtype=dtype,
        head_type=model_args.head_type,
        loss_type=model_args.loss_type,
        use_bias=model_args.use_bias,
        num_atoms=model_args.num_atoms,
        V_min=model_args.V_min,
        V_max=model_args.V_max,
    )
    if model_args.init_mode == "reuse" and model_args.head_type == "Q":
        classifier.reuse_init_head(model_args.classifier_model_id)
    elif model_args.init_mode == "zero":
        classifier.zero_init_head()
    return classifier


@torch.no_grad()
def evaluate(model, loader, accelerator):
    """Value-prediction quality on held-out prompts."""
    model.eval()
    losses, preds, labels = [], [], []
    unwrapped = accelerator.unwrap_model(model)
    for batch in loader:
        loss, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["rewards"],
            loss_mask=batch["loss_mask"],
            loss_weights=torch.ones_like(batch["loss_weights"]),
        )
        losses.append(loss)
        mask = batch["loss_mask"][:, 1:] if unwrapped.head_type == "Q" else batch["loss_mask"]
        predicted = unwrapped.calculate_predictions(logits)
        preds.append(predicted[mask])
        labels.append(batch["rewards"].unsqueeze(1).expand_as(mask)[mask])
    model.train()

    if not losses:
        return {}
    all_preds = accelerator.gather_for_metrics(torch.cat(preds))
    all_labels = accelerator.gather_for_metrics(torch.cat(labels))
    metrics = {
        "eval/loss": torch.mean(accelerator.gather(torch.stack(losses))).item(),
        "eval/explained_variance": explained_variance(all_preds, all_labels).item(),
        "eval/r2": r_squared(all_preds, all_labels).item(),
    }
    auc = roc_auc(all_preds, all_labels)
    if auc == auc:  # not NaN: both classes present
        metrics["eval/roc_auc"] = auc
        metrics["eval/accuracy"] = float(((all_preds > 0.5) == (all_labels > 0.5)).float().mean())
    return metrics


def train(model_args, data_args, training_args):
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)
    set_seed(training_args.seed)

    output_dir = training_args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(
                {k: vars(v) for k, v in
                 (("model", model_args), ("data", data_args), ("training", training_args))},
                f, indent=2, default=str,
            )

    ref_vocab_size = validate_pair(model_args.ref_model_id, model_args.classifier_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_args.ref_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_pairs(data_args.train_file)
    accelerator.print(f"Loaded {len(records)} (prompt, response, reward) rows from {data_args.train_file}")
    examples = build_examples(
        records, tokenizer,
        max_length=data_args.max_length,
        use_all_response_tokens=data_args.use_all_response_tokens,
    )
    train_data, eval_data = split_by_prompt(
        examples, data_args.eval_ratio, data_args.eval_max_size, seed=training_args.seed
    )
    accelerator.print(
        f"Training examples: {len(train_data['input_ids'])}, held-out: {len(eval_data['input_ids'])}"
    )

    collate_fn = lambda batch: collate(batch, pad_token_id=tokenizer.pad_token_id)  # noqa: E731
    train_loader = DataLoader(
        ClassifierDataset(train_data), batch_size=training_args.batch_size,
        shuffle=True, drop_last=True, collate_fn=collate_fn, pin_memory=True,
        num_workers=data_args.preprocessing_num_workers,
    )
    eval_loader = DataLoader(
        ClassifierDataset(eval_data), batch_size=training_args.batch_size,
        shuffle=False, collate_fn=collate_fn, pin_memory=True,
    )

    model = build_classifier(model_args, ref_vocab_size)
    if training_args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay
    )
    steps_per_epoch = len(train_loader) // training_args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, training_args.warmup_steps, max(1, steps_per_epoch * training_args.num_epochs)
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    tracking = training_args.wandb_project is not None
    if tracking:
        accelerator.init_trackers(
            training_args.wandb_project,
            config={"model": vars(model_args), "data": vars(data_args), "training": vars(training_args)},
            init_kwargs={"wandb": {"entity": training_args.wandb_entity, "name": training_args.run_name}},
        )

    step = 0
    model.train()
    for epoch in range(training_args.num_epochs):
        progress = tqdm(
            train_loader, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch}"
        )
        for batch in progress:
            with accelerator.accumulate(model):
                loss, _ = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["rewards"],
                    loss_mask=batch["loss_mask"],
                    loss_weights=batch["loss_weights"],
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if tracking:
                accelerator.log(
                    {"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0], "epoch": epoch},
                    step=step,
                )
            if training_args.eval_freq > 0 and step % training_args.eval_freq == 0:
                metrics = evaluate(model, eval_loader, accelerator)
                accelerator.print(f"step {step}: {metrics}")
                if tracking:
                    accelerator.log(metrics, step=step)
            if training_args.ckpt_freq > 0 and step % training_args.ckpt_freq == 0:
                save(model, tokenizer, accelerator, os.path.join(output_dir, f"ckpt_{step}"))

    metrics = evaluate(model, eval_loader, accelerator)
    accelerator.print(f"final: {metrics}")
    if tracking:
        accelerator.log(metrics, step=step)
        accelerator.end_training()
    save(model, tokenizer, accelerator, output_dir)
    return metrics


def save(model, tokenizer, accelerator, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(path)
        tokenizer.save_pretrained(path)
        accelerator.print(f"Saved classifier to {path}")
    accelerator.wait_for_everyone()
