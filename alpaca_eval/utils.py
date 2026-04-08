import json
import os
import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessorList
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from scipy.stats import entropy


def read_jsonl(path):
    results = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines) - 1, -1, -1):
        results.insert(0, json.loads(lines[i]))
        del lines[i]
    return results


def write_jsonl(results, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write('\n'.join(json.dumps(e) for e in results))


def write_json_array(results, path):
    """
    Writes a list of results to a file as a single JSON array.
    Args:
        results (list): List of serializable objects.
        path (str): Output file path.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(results, f)


def get_message(instruction):
    message = [
        {"role": "user", "content": instruction},
    ]
    return message


def tokenize_with_chat_template(tokenizer, prompts, use_chat_template, device):
    if use_chat_template:
        formatted_prompts = [tokenizer.apply_chat_template(get_message(prompt), add_generation_prompt=True, tokenize=False) for prompt in prompts]
        inputs = tokenizer(formatted_prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
    else:
        formatted_prompts = prompts
        inputs = tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(device)
    return inputs, formatted_prompts


def generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, inputs, generate_kwargs, return_output_only, return_text, eta):
    # we need to initialize the logit processor every time we generate
    logit_processor.reset_classifier_state()
    logit_processors = LogitsProcessorList([logit_processor])
    if eta !=0:
        with torch.no_grad():
            outputs = ref_model.generate(**inputs, logits_processor=logit_processors, pad_token_id=tokenizer.pad_token_id, **generate_kwargs)
    else:
        with torch.no_grad():
            outputs = ref_model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **generate_kwargs)
    if return_output_only:
        if isinstance(outputs, dict) and 'sequences' in outputs:
            outputs['sequences'] = outputs['sequences'][:, inputs['input_ids'].shape[1]:]
        else:
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
    if return_text:
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_outputs
    else:
        return outputs


def get_output_indices(outputs, eos_token_id):
    # assume there is at most one eos_token per sequence, otherwise will raise error
    outputs_end_indices_tuple = (outputs == eos_token_id).nonzero(as_tuple=True)
    assert len(outputs_end_indices_tuple[0]) <= outputs.shape[0], "There are multiple eos tokens in the same sequence.".format(outputs_end_indices_tuple)
    if len(outputs_end_indices_tuple[0]) < outputs.shape[0]:
        print('there exists some generation without eos token', outputs.shape)
        print(outputs[:, -1])
    seen_indices = []
    # default to the last token if no eos token is found
    outputs_end_indices = torch.ones((outputs.shape[0])).to(outputs_end_indices_tuple[0].device, dtype=torch.long) * (outputs.shape[1] - 1)
    for i in range(len(outputs_end_indices_tuple[0])):
        assert outputs_end_indices_tuple[0][i].item() not in seen_indices
        seen_indices.append(outputs_end_indices_tuple[0][i].item())
        outputs_end_indices[outputs_end_indices_tuple[0][i]] = outputs_end_indices_tuple[1][i]
    return outputs_end_indices


def create_classifier_data(all_data, use_all_ref_tokens, max_length=None):
    print("Creating classifier data...")
    classifier_data = {'input_ids': [], 'target_ids': [], 'rewards': [], 'loss_weights': []}#, 'num_token_full_response': []}
    prompt_key = 'prompt_tokenized'
    response_key = 'response_tokenized'
    num_token_prompt_key = 'num_response_tokens_in_partial_guided_prompts'
    reward_key = 'reward'
    assert use_all_ref_tokens in [0, 1]  # 2 needs to be handled with include rollin
    # input_ids will be roll-in, target_ids will be roll_out *sequence*
    for i in tqdm(range(len(all_data))):
        assert len(all_data[i][prompt_key]) == len(all_data[i][response_key]) == len(all_data[i][reward_key])
        loss_weight = 1
        for j in range(len(all_data[i][prompt_key])):
            input_ids = all_data[i][prompt_key][j][:-1]
            if use_all_ref_tokens == 0:
                target_ids = [all_data[i][prompt_key][j][-1]]
            else:
                target_ids = [all_data[i][prompt_key][j][-1]] + all_data[i][response_key][j]
            # num_token_full_response = all_data[i][num_token_prompt_key][j] + len(target_ids) - 1
            reward = all_data[i][reward_key][j]

            if len(target_ids) == 0:
                continue

            if max_length != -1:
                # ensure input_ids + target_ids <= max_length to prevent OOM.
                # if input_ids is already larger, then skip.
                # if input_ids is less, then optionally truncate target_ids.
                if len(input_ids) >= max_length - 1:
                    continue
                if len(input_ids) + len(target_ids) > max_length:
                    target_ids = target_ids[:max_length - len(input_ids)]
                    # num_token_full_response = all_data[i][num_token_prompt_key][j] + len(target_ids) - 1

            # print('input_ids', len(input_ids), 'target_ids', len(target_ids), 'sum', len(input_ids) + len(target_ids))
            classifier_data['input_ids'].append(input_ids)
            classifier_data['target_ids'].append(target_ids)
            classifier_data['rewards'].append(reward)
            classifier_data['loss_weights'].append(loss_weight)
            # classifier_data['num_token_full_response'].append(num_token_full_response)
    return classifier_data


class CustomClassifierDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}


def calculate_explained_variance(predictions, labels):
    explained_variance = 1 - torch.var(predictions - labels) / torch.var(labels)
    return explained_variance


def calculate_r2(predictions, labels):
    ss_res = torch.sum(torch.square(labels - predictions))
    ss_tot = torch.sum(torch.square(labels - torch.mean(labels)))
    r2 = 1 - ss_res / ss_tot
    return r2


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
        """
        Args:
            atoms (torch.Tensor): Support points (atoms) of shape (n_atoms,).
            pmfs (torch.Tensor): Probabilities of shape (bs, seqlen, n_atoms).
        """
        self.atoms = atoms  # Shape: (n_atoms,)
        self.n_atoms = atoms.shape[0]
        self.pmfs = torch.softmax(logits, dim=-1)  # Shape: (bs, seqlen, n_atoms)
        self.log_pmfs = torch.log_softmax(logits, dim=-1)
        if not torch.allclose(self.pmfs.sum(dim=-1), torch.tensor(1.0), atol=1e-5):
            raise ValueError("PMFs must sum to 1 along the last dimension.")

    def expected_value(self):
        """Compute the expected value of the distribution."""
        # Shape: (bs, seqlen)
        return torch.sum(self.pmfs * self.atoms, dim=-1)

    def variance(self):
        """Compute the variance of the distribution."""
        # E[Z]² and E[Z²]
        expected_value = self.expected_value()  # Shape: (bs, seqlen)
        expected_value_squared = expected_value ** 2
        expected_atoms_squared = torch.sum(self.pmfs * (self.atoms ** 2), dim=-1)
        # Variance = E[Z²] - (E[Z])²
        return expected_atoms_squared - expected_value_squared

    def entropy(self):
        """Compute the entropy of the categorical distribution."""
        # Shape: (bs, seqlen)
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
    # [bs, seqlen, vocab_size]
    log_p1 = torch.log_softmax(logits1, dim=-1)
    log_p2 = torch.log_softmax(logits2, dim=-1)
    p1 = torch.softmax(logits1, dim=-1)

    # Compute the KL terms: p1 * (log_p1 - log_p2)
    # If p1 == 0, we mask out that term to avoid NaN.
    # Where p1=0, (log_p1 - log_p2) is irrelevant and can be replaced by 0.
    kl_elements = (log_p1 - log_p2)
    kl_elements = torch.where(p1 > 0, kl_elements, torch.zeros_like(kl_elements))

    # Now sum over the vocabulary dimension
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
    # Ensure the path doesn't end with a slash
    normalized_path = path.rstrip("/")
    # Get the parent directory
    parent_dir = os.path.dirname(normalized_path)
    return parent_dir


def resolve_dict_value(d1, d2, key1, key2=None):
    # use key1 value from d1 if exists, otherwise use key2 value from d2
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
        unwrapped_model.save_pretrained(
            save_dir,
            push_to_hub=push_to_hub,
            repo_id=repo_id,
        )
        tokenizer.save_pretrained(
            save_dir,
            push_to_hub=push_to_hub,
            repo_id=repo_id,
        )
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, 'lr_scheduler.pt'))
        print(f"Model saved to {save_dir}.")
    accelerator.wait_for_everyone()


def custom_collate_fn(batch: list[dict[str, torch.Tensor]], pad_token_id: int):
    """
    keys: input_ids, target_ids, rewards, loss_weights
    """
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
