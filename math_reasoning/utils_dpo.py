import torch


class DpoDataset(torch.utils.data.Dataset):
    def __init__(self, all_data):
        self.data = []
        for i, ex in enumerate(all_data):
            # Print only the first example structure, not every example
            if i == 0:
                print("First example structure:")
                print("Available keys:", list(ex.keys()))
                print("Prompt type:", type(ex.get('prompt')))
                print("Fully guided predictions length:", len(ex.get('fully_guided_predictions', [])))
                print("Partial guided predictions length:", len(ex.get('partial_guided_predictions', [])))
                print("Correctness length:", len(ex.get('partial_guided_predictions_correctness', [])))
            
            prompt = ex['prompt']
            partial_guided_responses = ex['partial_guided_predictions']
            fully_guided_responses = ex['fully_guided_predictions']
            correctness = ex['partial_guided_predictions_correctness']

            # Since you mentioned the lengths will match, we can use the full length
            num_pairs = len(correctness)
            
            for j in range(num_pairs):
                full_response = fully_guided_responses[j]
                partial_response = partial_guided_responses[j]
                is_partial_correct = correctness[j]

                # If partial is correct (1), then partial is chosen, full is rejected
                # If partial is incorrect (0), then full is chosen, partial is rejected
                if is_partial_correct == 1:
                    chosen = partial_response
                    rejected = full_response
                else:
                    chosen = full_response
                    rejected = partial_response

                self.data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'label': float(is_partial_correct)  # Store the original correctness for evaluation
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dpo_collate_fn(batch, pad_token_id, tokenizer=None):
    prompts = [item['prompt'] for item in batch]
    chosen = [item['chosen'] for item in batch]
    rejected = [item['rejected'] for item in batch]
    labels = [item['label'] for item in batch]

    def build_input(prompt, response, tokenizer):
        # Handle both string and pre-tokenized inputs
        if isinstance(prompt, str) and isinstance(response, str):
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided when inputs are strings")
            # Tokenize the full conversation
            full_text = prompt + response
            tokenized = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized.input_ids.squeeze(0).tolist()
            attention_mask = [1] * len(input_ids)
        elif isinstance(prompt, list) and isinstance(response, list):
            # Already tokenized
            input_ids = prompt + response
            attention_mask = [1] * len(input_ids)
        else:
            raise ValueError(f"Mismatched input types: prompt type {type(prompt)}, response type {type(response)}")
        
        return input_ids, attention_mask

    # Build inputs for chosen and rejected responses
    input_chosen, mask_chosen = zip(*[build_input(p, c, tokenizer) for p, c in zip(prompts, chosen)])
    input_rejected, mask_rejected = zip(*[build_input(p, r, tokenizer) for p, r in zip(prompts, rejected)])

    def pad_sequences(seq_list, pad_id):
        if not seq_list:
            return torch.tensor([])
        max_len = max(len(seq) for seq in seq_list)
        padded = []
        for seq in seq_list:
            padded_seq = seq + [pad_id] * (max_len - len(seq))
            padded.append(padded_seq)
        return torch.tensor(padded, dtype=torch.long)

    return {
        'input_ids_chosen': pad_sequences(input_chosen, pad_token_id),
        'attention_mask_chosen': pad_sequences(mask_chosen, 0),
        'input_ids_rejected': pad_sequences(input_rejected, pad_token_id),
        'attention_mask_rejected': pad_sequences(mask_rejected, 0),
        'labels': torch.tensor(labels, dtype=torch.float)
    }


def compute_log_probs(model, input_ids, attention_mask, training=True):
    """Compute log probabilities for the sequence"""
    # Only use no_grad when not training or when explicitly specified
    if not training:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get logits for all positions except the last one
    logits = outputs.logits[:, :-1, :].contiguous()
    # Get labels (shifted input_ids)
    labels = input_ids[:, 1:].contiguous()
    
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Gather log probabilities for the actual tokens
    gathered_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out padding tokens - use the shifted attention mask
    mask = attention_mask[:, 1:].contiguous().float()
    masked_log_probs = gathered_log_probs * mask
    
    # Sum log probabilities for each sequence (only count non-padded tokens)
    sequence_log_probs = masked_log_probs.sum(dim=1)
    
    return sequence_log_probs


def compute_log_probs_with_reference(model, ref_model, input_ids, attention_mask):
    """Compute log probabilities with reference model for DPO"""
    model_log_probs = compute_log_probs(model, input_ids, attention_mask, training=True)
    ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, training=False)
    return model_log_probs, ref_log_probs


def dpo_loss(model, batch, beta=0.1, ref_model=None):
    """
    Compute DPO loss
    
    Args:
        model: The model being trained (prepared by accelerator)
        batch: Batch of data containing chosen and rejected responses
        beta: Temperature parameter for DPO
        ref_model: Reference model (not prepared by accelerator, manually placed on device)
    
    Returns:
        loss: DPO loss
        logp_chosen: Log probabilities for chosen responses
        logp_rejected: Log probabilities for rejected responses
    """
    # Accelerator handles device placement for prepared models and data
    input_ids_chosen = batch['input_ids_chosen']
    attention_mask_chosen = batch['attention_mask_chosen']
    input_ids_rejected = batch['input_ids_rejected']
    attention_mask_rejected = batch['attention_mask_rejected']
    
    if ref_model is not None:
        # Compute log probabilities for both model and reference
        logp_chosen_model = compute_log_probs(model, input_ids_chosen, attention_mask_chosen, training=True)
        logp_rejected_model = compute_log_probs(model, input_ids_rejected, attention_mask_rejected, training=True)
        
        # For reference model, we need to move data to its device
        device = next(ref_model.parameters()).device
        with torch.no_grad():
            logp_chosen_ref = compute_log_probs(
                ref_model, 
                input_ids_chosen.to(device), 
                attention_mask_chosen.to(device),
                training=False
            )
            logp_rejected_ref = compute_log_probs(
                ref_model, 
                input_ids_rejected.to(device), 
                attention_mask_rejected.to(device),
                training=False
            )
        
        # Move reference logprobs back to the main model's device
        logp_chosen_ref = logp_chosen_ref.to(logp_chosen_model.device)
        logp_rejected_ref = logp_rejected_ref.to(logp_rejected_model.device)
        
        # DPO loss with reference model
        logits = beta * ((logp_chosen_model - logp_chosen_ref) - (logp_rejected_model - logp_rejected_ref))
    else:
        # Simplified version without explicit reference model
        logp_chosen_model = compute_log_probs(model, input_ids_chosen, attention_mask_chosen, training=True)
        logp_rejected_model = compute_log_probs(model, input_ids_rejected, attention_mask_rejected, training=True)
        
        logits = beta * (logp_chosen_model - logp_rejected_model)
    
    # DPO loss: -log(sigmoid(logits))
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    
    return loss, logp_chosen_model, logp_rejected_model


def compute_dpo_accuracy(model, batch, beta=0.1, ref_model=None):
    """Compute accuracy for DPO model"""
    input_ids_chosen = batch['input_ids_chosen']
    attention_mask_chosen = batch['attention_mask_chosen']
    input_ids_rejected = batch['input_ids_rejected']
    attention_mask_rejected = batch['attention_mask_rejected']
    
    with torch.no_grad():
        if ref_model is not None:
            logp_chosen_model = compute_log_probs(model, input_ids_chosen, attention_mask_chosen, training=False)
            logp_rejected_model = compute_log_probs(model, input_ids_rejected, attention_mask_rejected, training=False)
            
            # For reference model, we need to move data to its device
            device = next(ref_model.parameters()).device
            logp_chosen_ref = compute_log_probs(
                ref_model, 
                input_ids_chosen.to(device), 
                attention_mask_chosen.to(device),
                training=False
            )
            logp_rejected_ref = compute_log_probs(
                ref_model, 
                input_ids_rejected.to(device), 
                attention_mask_rejected.to(device),
                training=False
            )
            
            # Move reference logprobs back to the main model's device
            logp_chosen_ref = logp_chosen_ref.to(logp_chosen_model.device)
            logp_rejected_ref = logp_rejected_ref.to(logp_rejected_model.device)
            
            logits = beta * ((logp_chosen_model - logp_chosen_ref) - (logp_rejected_model - logp_rejected_ref))
        else:
            logp_chosen_model = compute_log_probs(model, input_ids_chosen, attention_mask_chosen, training=False)
            logp_rejected_model = compute_log_probs(model, input_ids_rejected, attention_mask_rejected, training=False)
            
            logits = beta * (logp_chosen_model - logp_rejected_model)
        
        # Accuracy: how often the model prefers chosen over rejected
        preferences = (logits > 0).float()
        accuracy = preferences.mean()
    
    return accuracy.item()