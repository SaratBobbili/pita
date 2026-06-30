import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


def read_jsonl(path):
    results = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines) - 1, -1, -1):
        results.insert(0, json.loads(lines[i]))
        del lines[i]
    return results


def _to_native(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, list):
        return [_to_native(x) for x in v]
    return v


def read_parquet_records(path):
    import pandas as pd
    df = pd.read_parquet(path)
    return [{k: _to_native(v) for k, v in row.items()} for row in df.to_dict('records')]


def write_jsonl(results, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write('\n'.join(json.dumps(e) for e in results))


def write_json_array(results, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(results, f)


def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def tokenize_with_chat_template(tokenizer, prompts, use_chat_template, device):
    if use_chat_template:
        formatted_prompts = [tokenizer.apply_chat_template(get_message(prompt), add_generation_prompt=True, tokenize=False) for prompt in prompts]
        inputs = tokenizer(formatted_prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
    else:
        formatted_prompts = prompts
        inputs = tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(device)
    return inputs, formatted_prompts


def get_output_indices(outputs, eos_token_id):
    outputs_end_indices_tuple = (outputs == eos_token_id).nonzero(as_tuple=True)
    assert len(outputs_end_indices_tuple[0]) <= outputs.shape[0], "There are multiple eos tokens in the same sequence."
    if len(outputs_end_indices_tuple[0]) < outputs.shape[0]:
        print('there exists some generation without eos token', outputs.shape)
        print(outputs[:, -1])
    seen_indices = []
    outputs_end_indices = torch.ones((outputs.shape[0])).to(outputs_end_indices_tuple[0].device, dtype=torch.long) * (outputs.shape[1] - 1)
    for i in range(len(outputs_end_indices_tuple[0])):
        assert outputs_end_indices_tuple[0][i].item() not in seen_indices
        seen_indices.append(outputs_end_indices_tuple[0][i].item())
        outputs_end_indices[outputs_end_indices_tuple[0][i]] = outputs_end_indices_tuple[1][i]
    return outputs_end_indices


def create_classifier_data(all_data, use_all_ref_tokens, max_length=None):
    print("Creating classifier data...")
    classifier_data = {'input_ids': [], 'target_ids': [], 'rewards': [], 'loss_weights': []}
    prompt_key = 'prompt_tokenized'
    response_key = 'response_tokenized'
    reward_key = 'reward'
    assert use_all_ref_tokens in [0, 1]
    for i in tqdm(range(len(all_data))):
        assert len(all_data[i][prompt_key]) == len(all_data[i][response_key]) == len(all_data[i][reward_key])
        loss_weight = 1
        for j in range(len(all_data[i][prompt_key])):
            input_ids = all_data[i][prompt_key][j][:-1]
            if use_all_ref_tokens == 0:
                target_ids = [all_data[i][prompt_key][j][-1]]
            else:
                target_ids = [all_data[i][prompt_key][j][-1]] + all_data[i][response_key][j]
            reward = all_data[i][reward_key][j]

            if len(target_ids) == 0:
                continue

            if max_length != -1:
                if len(input_ids) >= max_length - 1:
                    continue
                if len(input_ids) + len(target_ids) > max_length:
                    target_ids = target_ids[:max_length - len(input_ids)]

            classifier_data['input_ids'].append(input_ids)
            classifier_data['target_ids'].append(target_ids)
            classifier_data['rewards'].append(reward)
            classifier_data['loss_weights'].append(loss_weight)
    return classifier_data


class CustomClassifierDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}


def calculate_explained_variance(predictions, labels):
    return 1 - torch.var(predictions - labels) / torch.var(labels)


def calculate_r2(predictions, labels):
    ss_res = torch.sum(torch.square(labels - predictions))
    ss_tot = torch.sum(torch.square(labels - torch.mean(labels)))
    return 1 - ss_res / ss_tot


def get_average_reward(all_data, eval_key, simulation_rounds):
    average_rewards = []
    for i in range(simulation_rounds):
        rewards = []
        for j in range(len(all_data)):
            random_idx = np.random.randint(len(all_data[j][eval_key]))
            rewards.append(all_data[j][eval_key][random_idx])
        average_rewards.append(np.mean(rewards))
    return average_rewards


class CategoricalDistributionRL:
    def __init__(self, atoms, logits):
        self.atoms = atoms
        self.n_atoms = atoms.shape[0]
        self.pmfs = torch.softmax(logits, dim=-1)
        self.log_pmfs = torch.log_softmax(logits, dim=-1)
        if not torch.allclose(self.pmfs.sum(dim=-1), torch.tensor(1.0), atol=1e-5):
            raise ValueError("PMFs must sum to 1 along the last dimension.")

    def expected_value(self):
        return torch.sum(self.pmfs * self.atoms, dim=-1)

    def variance(self):
        expected_value = self.expected_value()
        expected_value_squared = expected_value ** 2
        expected_atoms_squared = torch.sum(self.pmfs * (self.atoms ** 2), dim=-1)
        return expected_atoms_squared - expected_value_squared

    def entropy(self):
        return -torch.sum(self.pmfs * self.log_pmfs, dim=-1)


def calculate_mle_stats(logits, atoms):
    assert len(logits.shape) == 3
    assert atoms.shape == (logits.size(-1),)
    pmfs = torch.softmax(logits, dim=-1)
    dist = CategoricalDistributionRL(atoms, pmfs)
    return {
        'expected_value': dist.expected_value(),
        'variance': dist.variance(),
        'entropy': dist.entropy()
    }


def kl_divergence(logits1, logits2):
    assert logits1.shape == logits2.shape, f"Shapes of logits1 and logits2 must match: {logits1.shape} vs. {logits2.shape}"
    assert len(logits1.shape) == 3, f"Expected 3D logits, got {logits1.shape}"
    log_p1 = torch.log_softmax(logits1, dim=-1)
    log_p2 = torch.log_softmax(logits2, dim=-1)
    p1 = torch.softmax(logits1, dim=-1)
    kl_elements = (log_p1 - log_p2)
    kl_elements = torch.where(p1 > 0, kl_elements, torch.zeros_like(kl_elements))
    kl = torch.sum(p1 * kl_elements, dim=-1)
    return kl


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, max_batch_size, max_tokens_per_batch, shuffle):
        super(DynamicBatchSampler, self).__init__(dataset)
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        batch = []
        batch_max_length = 0
        for idx in indices:
            current_item = self.dataset[idx]
            current_num_tokens = len(current_item['input_ids']) + len(current_item['target_ids'])
            if current_num_tokens > batch_max_length:
                batch_max_length = current_num_tokens
            total_tokens = batch_max_length * (len(batch) + 1)
            if total_tokens > self.max_tokens_per_batch or len(batch) == self.max_batch_size:
                assert len(batch) > 0, 'effective batch size is 0, max_tokens_per_batch of {0} is too small for {1} tokens'.format(self.max_tokens_per_batch, current_num_tokens)
                yield batch
                batch = []
                batch_max_length = current_num_tokens
            batch.append(idx)
        if len(batch) > 0:
            yield batch


def get_parent_directory(path):
    normalized_path = path.rstrip("/")
    return os.path.dirname(normalized_path)


def resolve_dict_value(d1, d2, key1, key2=None):
    if key2 is None:
        key2 = key1
    if key1 in d1 and d1[key1] is not None:
        return d1[key1]
    else:
        return d2[key2]


def save_model(model, tokenizer, optimizer, lr_scheduler, accelerator, save_dir, push_to_hub=False, repo_id=None):
    if push_to_hub:
        assert repo_id is not None, "repo_id must be provided if push_to_hub is True."
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(save_dir, push_to_hub=push_to_hub, repo_id=repo_id)
        tokenizer.save_pretrained(save_dir, push_to_hub=push_to_hub, repo_id=repo_id)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, 'lr_scheduler.pt'))
        print(f"Model saved to {save_dir}.")
    accelerator.wait_for_everyone()


def custom_collate_fn(batch: list[dict[str, torch.Tensor]], pad_token_id: int):
    max_batch_length = max([len(x['input_ids']) + len(x['target_ids']) for x in batch])
    padded_seq = []
    attention_mask = []
    loss_mask = []
    for x in batch:
        padding_len = max_batch_length - len(x['input_ids']) - len(x['target_ids'])
        padded_seq.append(torch.cat([
            torch.full((padding_len,), pad_token_id, dtype=torch.long),
            torch.tensor(x['input_ids'], dtype=torch.long),
            torch.tensor(x['target_ids'], dtype=torch.long)
        ]))
        attention_mask.append(torch.cat([
            torch.zeros(padding_len, dtype=torch.bool),
            torch.ones(len(x['input_ids']) + len(x['target_ids']), dtype=torch.bool)
        ]))
        loss_mask.append(torch.cat([
            torch.zeros(padding_len + len(x['input_ids']), dtype=torch.bool),
            torch.ones(len(x['target_ids']), dtype=torch.bool)
        ]))
    padded_seq = torch.stack(padded_seq)
    attention_mask = torch.stack(attention_mask)
    loss_mask = torch.stack(loss_mask)
    return {
        'input_ids': padded_seq,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'rewards': torch.tensor([x['rewards'] for x in batch]).float(),
        'loss_weights': torch.tensor([x['loss_weights'] for x in batch]).float()
    }


def perplexity_with_classifier_guidance(ref_model, tokenizer, logit_processor, inputs, response_inputs, eta):
    _ = eta, tokenizer
    assert inputs["input_ids"].shape[0] == response_inputs["input_ids"].shape[0]
    device = inputs["input_ids"].device
    response_ids = response_inputs["input_ids"]
    response_mask = response_inputs.get("attention_mask")
    if response_mask is None:
        response_mask = torch.ones_like(response_ids)
    batch_size = inputs["input_ids"].shape[0]
    ppls = []
    for b in tqdm(range(batch_size), desc="guided_ppl"):
        logit_processor.reset_classifier_state()
        prompt_row = inputs["input_ids"][b : b + 1]
        prompt_attn = inputs["attention_mask"][b : b + 1]
        rm = response_mask[b].bool()
        resp_real = response_ids[b][rm]
        rlen = int(resp_real.numel())
        if rlen == 0:
            ppls.append(torch.tensor(float("nan"), device=device))
            continue
        total_nll = torch.zeros((), device=device, dtype=torch.float32)
        for t in range(rlen):
            if t == 0:
                prefix_ids = prompt_row
                prefix_attn = prompt_attn
            else:
                gen_piece = resp_real[:t].unsqueeze(0)
                prefix_ids = torch.cat([prompt_row, gen_piece], dim=1)
                gen_attn = torch.ones(1, t, device=device, dtype=prompt_attn.dtype)
                prefix_attn = torch.cat([prompt_attn, gen_attn], dim=1)
            with torch.no_grad():
                out = ref_model(input_ids=prefix_ids, attention_mask=prefix_attn)
            ref_logits = out.logits[:, -1, :]
            guided = logit_processor(prefix_ids, ref_logits)
            tok = resp_real[t]
            log_prob = F.log_softmax(guided.float(), dim=-1)[0, tok]
            total_nll = total_nll - log_prob
        ppls.append(torch.exp(total_nll / rlen))
    return torch.stack(ppls)
