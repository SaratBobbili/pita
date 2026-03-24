#!/usr/bin/env python3
"""
Train a Llama-3-1B reward model with BCE loss on preference data.

Input  = prompt + partially_guided_prediction vs prompt + fully_guided_prediction pairs
Target = partial_guided_predictions_correctness (0 / 1), indicating preference

Example usage:
python train_reward_model.py \
  --data_path /mnt/research/.../all_train_pref_data_binary.jsonl \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./reward_model
"""
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path", type=str, required=True, help="JSONL with preference data"
    )
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--output_dir", type=str, default="./reward_model")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument(
        "--eval_holdout_frac",
        type=float,
        default=0.02,
        help="Fraction for validation split (0 → no eval)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    # --------------------------------------------------------------------- #
    # 1. Load raw dataset
    # --------------------------------------------------------------------- #
    logger.info("Loading dataset from %s", args.data_path)
    raw = load_dataset("json", data_files={"train": args.data_path})["train"]

    # --------------------------------------------------------------------- #
    # 2. Flatten lists → each row is a (prompt+completion pair, preference label)
    # --------------------------------------------------------------------- #
    def explode(batch):
        out_chosen, out_rejected, out_label = [], [], []
        for prompt, partial_preds, full_preds, partial_labels in zip(
            batch["prompt"],
            batch["partial_guided_predictions"],
            batch["fully_guided_predictions"],
            batch["partial_guided_predictions_correctness"],
        ):
            for partial, full, is_partial_preferred in zip(partial_preds, full_preds, partial_labels):
                if is_partial_preferred == 1:
                    chosen = partial
                    rejected = full
                    label = 1.0  # partial preferred
                else:
                    chosen = full
                    rejected = partial
                    label = 0.0  # fully guided preferred
                out_chosen.append(prompt + "\n" + chosen)
                out_rejected.append(prompt + "\n" + rejected)
                out_label.append(label)
        return {"chosen": out_chosen, "rejected": out_rejected, "label": out_label}

    flat_ds = raw.map(
        explode,
        batched=True,
        remove_columns=raw.column_names,
        desc="Creating preference pairs from partial vs fully guided completions",
    )

    logger.info("Dataset size after preference conversion: %d", len(flat_ds))

    # --------------------------------------------------------------------- #
    # 3. Tokeniser + pad token fix
    # --------------------------------------------------------------------- #
    logger.info("Loading tokenizer %s", args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:  # Llama-3 trick
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def preprocess(batch):
        chosen_enc = tok(
            batch["chosen"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        rejected_enc = tok(
            batch["rejected"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "label": batch["label"],
        }

    proc_ds = flat_ds.map(
        preprocess,
        batched=True,
        remove_columns=["chosen", "rejected"],
        desc="Tokenizing preference pairs",
    )
    # Optional train/val split
    if 0 < args.eval_holdout_frac < 1:
        split = proc_ds.train_test_split(
            test_size=args.eval_holdout_frac, seed=args.seed
        )
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = proc_ds, None

    # --------------------------------------------------------------------- #
    # 4. Model wrapper for pairwise scoring
    # --------------------------------------------------------------------- #
    class RewardModelWrapper(nn.Module):
        def __init__(self, base_model_name):
            super().__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=1
            )

        def forward(
            self,
            chosen_input_ids,
            chosen_attention_mask,
            rejected_input_ids,
            rejected_attention_mask,
            labels=None,  # Accept labels parameter but don't use it
        ):
            chosen_outputs = self.model(
                input_ids=chosen_input_ids, attention_mask=chosen_attention_mask
            )
            rejected_outputs = self.model(
                input_ids=rejected_input_ids, attention_mask=rejected_attention_mask
            )

            chosen_scores = chosen_outputs.logits.view(-1)
            rejected_scores = rejected_outputs.logits.view(-1)
            return chosen_scores, rejected_scores

    model = RewardModelWrapper(args.model_name)

    if model.model.config.pad_token_id is None:
        model.model.config.pad_token_id = tok.pad_token_id
    model.model.resize_token_embeddings(len(tok))

    # --------------------------------------------------------------------- #
    # 5. Custom Trainer with pairwise loss
    # --------------------------------------------------------------------- #
    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            #print("Available keys in inputs:", list(inputs.keys()))
            labels = inputs.pop("labels").float()  # shape (batch,)
            chosen_scores, rejected_scores = model(
                chosen_input_ids=inputs.pop("chosen_input_ids"),
                chosen_attention_mask=inputs.pop("chosen_attention_mask"),
                rejected_input_ids=inputs.pop("rejected_input_ids"),
                rejected_attention_mask=inputs.pop("rejected_attention_mask"),
            )
            diff = chosen_scores - rejected_scores
            loss = F.binary_cross_entropy_with_logits(diff, labels)
            return (loss, (chosen_scores, rejected_scores)) if return_outputs else loss

    # Add evaluation metrics function for accuracy of preferences
    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0] - logits[1]  # difference of chosen and rejected scores
        logits = logits.reshape(-1)
        labels = eval_pred.label_ids.reshape(-1)
        preds = (logits > 0).astype(int)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    # --------------------------------------------------------------------- #
    # 6. TrainingArguments
    # --------------------------------------------------------------------- #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=500 if eval_ds else None,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="accuracy" if eval_ds else None,
        greater_is_better=True if eval_ds else None,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        fp16=False,
        bf16=torch.cuda.is_available(),
    )

    # --------------------------------------------------------------------- #
    # 7. Trainer & run
    # --------------------------------------------------------------------- #
    logger.info("Initialising Trainer")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if eval_ds else None,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
