import json
import os
import time
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed, get_constant_schedule_with_warmup
from functools import partial

from models.classifier import get_classifier_class
from training.dataset import (
    CustomClassifierDataset, DynamicBatchSampler, custom_collate_fn,
    calculate_explained_variance, calculate_r2, calculate_mle_stats, save_model
)
from training.builder import build_training_data


def run_training(cfg):
    accelerator = Accelerator()
    set_seed(cfg.training.seed * 42 + accelerator.process_index * 100003)

    output_dir = cfg.trainer.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        from omegaconf import OmegaConf
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.ref_model_id)
    classifier_tokenizer = AutoTokenizer.from_pretrained(cfg.models.classifier_model_id)
    assert len(tokenizer) == len(classifier_tokenizer), "tokenizer vocab size mismatch"
    vocab_size = AutoConfig.from_pretrained(cfg.models.classifier_model_id).vocab_size
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_classifier_data, id_eval_classifier_data = build_training_data(cfg)

    train_dataset = CustomClassifierDataset(train_classifier_data)
    id_eval_dataset = CustomClassifierDataset(id_eval_classifier_data)
    collate_fn = partial(custom_collate_fn, pad_token_id=tokenizer.pad_token_id)

    batch_size = cfg.training.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    id_eval_loader = DataLoader(id_eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, pin_memory=True)

    model_loading_kwargs = {}
    if cfg.models.dtype == 'bfloat16':
        model_loading_kwargs['torch_dtype'] = torch.bfloat16

    classifier_ckpt_path = cfg.models.classifier_ckpt_path or cfg.models.classifier_model_id
    classifier_model = get_classifier_class(cfg.models.classifier_arch).from_pretrained(
        classifier_ckpt_path, **model_loading_kwargs,
        num_labels=vocab_size, loss_type=cfg.models.loss_type,
        use_bias=bool(cfg.models.use_bias), classifier_type=cfg.models.classifier_type,
        device_map=device, num_atoms=11, V_min=0.0, V_max=1.0)

    if cfg.models.classifier_type == 'Q' and cfg.models.init_mode == 'zero':
        classifier_model.zero_init_classifier()
    elif cfg.models.classifier_type == 'Q' and cfg.models.init_mode == 'reuse':
        temp_model = AutoModelForCausalLM.from_pretrained(cfg.models.classifier_model_id, **model_loading_kwargs, device_map='cpu')
        lm_head_parameters = list(temp_model.lm_head.parameters())
        assert len(lm_head_parameters) == 1
        lm_head_weight = lm_head_parameters[0].data.to(device)
        if cfg.models.loss_type == "mle":
            classifier_model.score.weight.data = lm_head_weight.repeat(1, 11).view(vocab_size * 11, -1)
        else:
            classifier_model.score.weight.data = lm_head_weight
        del temp_model
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg.training.warmup_steps)

    gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
    num_epochs = cfg.training.num_epochs
    eval_freq = cfg.training.eval_freq
    ckpt_freq = cfg.training.ckpt_freq

    global_step = 0
    start_time = time.time()
    accumulated_loss = torch.tensor(0.0).to(device)

    use_wandb = cfg.wandb.enabled and accelerator.is_local_main_process
    if use_wandb:
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.run_name, config=dict(cfg))

    classifier_model.eval()
    save_dir = os.path.join(output_dir, 'ckpt_%d' % global_step)
    save_model(classifier_model, tokenizer, None, None, accelerator, save_dir=save_dir)

    classifier_model.train()
    classifier_model, optimizer, train_loader, id_eval_loader, scheduler = \
        accelerator.prepare(classifier_model, optimizer, train_loader, id_eval_loader, scheduler)

    for epoch in range(num_epochs):
        if use_wandb:
            wandb.log({'Epoch': epoch}, step=global_step)
        bar = tqdm(train_loader) if accelerator.is_local_main_process else train_loader
        torch.cuda.empty_cache()
        for batch_input_data in bar:
            global_step += accelerator.num_processes
            outputs = classifier_model(
                input_ids=batch_input_data['input_ids'],
                attention_mask=batch_input_data['attention_mask'],
                labels=batch_input_data['rewards'],
                loss_mask=batch_input_data['loss_mask'],
                loss_weights=batch_input_data['loss_weights'])
            loss = outputs.loss / gradient_accumulation_steps
            del batch_input_data, outputs
            torch.cuda.empty_cache()

            accelerator.backward(loss)
            accumulated_loss += loss.detach()

            if (global_step // accelerator.num_processes + 1) % gradient_accumulation_steps == 0:
                if cfg.models.classifier_type == "V":
                    accelerator.clip_grad_norm_(classifier_model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                if use_wandb:
                    wandb.log({
                        'Training Loss': accelerator.gather(accumulated_loss.unsqueeze(0)).mean().item(),
                        'Learning Rate': scheduler.get_last_lr()[0],
                    }, step=global_step)
                accumulated_loss = torch.tensor(0.0).to(device)

            if eval_freq != -1 and (global_step % eval_freq == 0 or global_step == 1):
                classifier_model.eval()
                unwrapped = classifier_model.module if hasattr(classifier_model, "module") else classifier_model
                eval_losses = []
                eval_predictions_list = []
                eval_labels_list = []

                with torch.no_grad():
                    for batch_input_data in id_eval_loader:
                        batch_input_data['loss_weights'] = torch.ones_like(batch_input_data['loss_weights'])
                        outputs = classifier_model(
                            input_ids=batch_input_data['input_ids'],
                            attention_mask=batch_input_data['attention_mask'],
                            labels=batch_input_data['rewards'],
                            loss_mask=batch_input_data['loss_mask'],
                            loss_weights=batch_input_data['loss_weights'])
                        eval_losses.append(outputs.loss)
                        cur_preds = unwrapped.calculate_predictions(outputs.logits)
                        loss_mask = batch_input_data['loss_mask']
                        if cfg.models.classifier_type == "Q":
                            cur_preds[~loss_mask[:, 1:]] = -1
                        else:
                            cur_preds[~loss_mask] = -1
                        eval_predictions_list.append(cur_preds)
                        seqlen = cur_preds.shape[1]
                        cur_labels = batch_input_data['rewards'].unsqueeze(1).repeat(1, seqlen)
                        if cfg.models.classifier_type == "Q":
                            cur_labels[~loss_mask[:, 1:]] = -1
                        else:
                            cur_labels[~loss_mask] = -1
                        eval_labels_list.append(cur_labels)
                        del batch_input_data, outputs
                        torch.cuda.empty_cache()

                avg_eval_loss = torch.mean(accelerator.gather(torch.stack(eval_losses)))
                all_preds = torch.cat([v.flatten() for v in eval_predictions_list])
                all_preds = all_preds[all_preds != -1]
                all_labels = torch.cat([v.flatten() for v in eval_labels_list])
                all_labels = all_labels[all_labels != -1]

                if use_wandb:
                    wandb.log({
                        'ID Eval Loss': avg_eval_loss.item(),
                        'ID Eval Explained Variance': calculate_explained_variance(all_preds, all_labels).item(),
                        'ID Eval R^2': calculate_r2(all_preds, all_labels).item(),
                    }, step=global_step)

                if cfg.models.inference_mode == 'bernoulli' and len(all_preds) > 0:
                    rounded = (all_preds > 0.5).float().cpu().numpy()
                    labels_np = all_labels.cpu().numpy()
                    accuracy = np.mean(rounded == labels_np)
                    if len(np.unique(labels_np)) > 1:
                        roc_auc = roc_auc_score(labels_np, all_preds.cpu().numpy())
                    else:
                        roc_auc = 0.0
                    if use_wandb:
                        wandb.log({'ID Eval Accuracy': accuracy, 'ID Eval ROC-AUC': roc_auc}, step=global_step)

                classifier_model.train()
                del eval_losses, eval_predictions_list, eval_labels_list
                torch.cuda.empty_cache()

            if ckpt_freq != -1 and global_step % ckpt_freq == 0:
                classifier_model.eval()
                save_dir = os.path.join(output_dir, 'ckpt_%d' % global_step)
                save_model(classifier_model, tokenizer, None, None, accelerator, save_dir=save_dir)
                classifier_model.train()
