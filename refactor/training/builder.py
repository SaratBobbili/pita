import json
import copy
import numpy as np
from training.dataset import read_parquet_records, create_classifier_data


def build_arithmetic_data(cfg, train_eval_problems_d):
    """Build all_data for arithmetic datasets (GSM8K/MATH) from collected parquet/jsonl."""
    data_path = cfg.data.train_file
    current_data = read_parquet_records(data_path)

    problem_position_d = {}
    all_data = []
    merge_keys = ['fully_guided_predictions', 'fully_guided_predictions_correctness', 'partial_guided_prompts',
                  'partial_guided_prompts_tokenized', 'num_response_tokens_in_partial_guided_prompts',
                  'partial_guided_responses_tokenized', 'partial_guided_predictions',
                  'partial_guided_predictions_correctness']

    for i in range(len(current_data)):
        current_problem = current_data[i]['problem']
        if current_problem not in problem_position_d:
            problem_position_d[current_problem] = len(all_data)
            all_data.append(current_data[i])
        else:
            for k in merge_keys:
                if k in current_data[i]:
                    all_data[problem_position_d[current_problem]][k].extend(current_data[i][k])

    reward_key = 'partial_guided_predictions_correctness'
    drop_no_variation = cfg.data.drop_no_variation

    for i in range(len(all_data)):
        current_rewards = []
        for j in range(len(all_data[i][reward_key])):
            current_reward = int(all_data[i][reward_key][j])
            current_rewards.append(current_reward)
        all_data[i]['reward'] = current_rewards

    if drop_no_variation:
        all_data = [d for d in all_data if len(set(d['reward'])) > 1]

    return all_data, train_eval_problems_d


def build_preference_data(cfg, train_eval_problems_d):
    """Build all_data for preference datasets (HH-RLHF, AlpacaEval) from collected jsonl."""
    data_path = cfg.data.train_file
    current_data = read_parquet_records(data_path)

    problem_position_d = {}
    all_data = []
    merge_keys = ['fully_guided_predictions', 'fully_guided_predictions_correctness', 'partial_guided_prompts',
                  'partial_guided_prompts_tokenized', 'num_response_tokens_in_partial_guided_prompts',
                  'partial_guided_responses_tokenized', 'partial_guided_predictions',
                  'partial_guided_predictions_correctness', 'fully_guided_responses_tokenized']

    for i in range(len(current_data)):
        current_problem = current_data[i]['problem']
        if current_problem not in problem_position_d:
            problem_position_d[current_problem] = len(all_data)
            all_data.append(current_data[i])
        else:
            for k in merge_keys:
                if k in current_data[i]:
                    all_data[problem_position_d[current_problem]][k].extend(current_data[i][k])

    partial_guided_data = []
    fully_guided_data = []
    for i in range(len(all_data)):
        common_info = {
            'prompt': all_data[i]['prompt'],
            'prompt_tokenized': all_data[i]['partial_guided_prompts_tokenized'],
            'num_response_tokens_in_partial_guided_prompts': -1,
            'problem': all_data[i]['problem']
        }
        partial_guided_sample = copy.deepcopy(common_info)
        fully_guided_sample = copy.deepcopy(common_info)
        partial_guided_sample['response_tokenized'] = all_data[i]['partial_guided_responses_tokenized']
        fully_guided_sample['response_tokenized'] = all_data[i]['fully_guided_responses_tokenized']
        partial_guided_sample['reward'] = [int(all_data[i]['partial_guided_predictions_correctness'][0])]
        fully_guided_sample['reward'] = [int(all_data[i]['fully_guided_predictions_correctness'][0])]
        partial_guided_data.append(partial_guided_sample)
        fully_guided_data.append(fully_guided_sample)

    all_data = partial_guided_data + fully_guided_data
    return all_data, train_eval_problems_d


def build_training_data(cfg):
    """Main entry: build classifier training data from collected generation outputs."""
    train_eval_save_path = cfg.data.train_eval_save_path or cfg.dataset.train_eval_save_path
    with open(train_eval_save_path, 'r') as f:
        train_eval_problems_d = json.load(f)

    if cfg.dataset.family == "arithmetic":
        all_data, train_eval_problems_d = build_arithmetic_data(cfg, train_eval_problems_d)
    else:
        all_data, train_eval_problems_d = build_preference_data(cfg, train_eval_problems_d)

    all_train_data = []
    all_eval_data = []
    for i in range(len(all_data)):
        problem_key = all_data[i].get('problem', all_data[i].get('prompt'))
        if problem_key in train_eval_problems_d:
            split = train_eval_problems_d[problem_key]['split']
        else:
            split = 'train'
        if split == 'train':
            all_train_data.append(all_data[i])
        else:
            all_eval_data.append(all_data[i])

    print(f'Training problems: {len(all_train_data)}, Eval problems: {len(all_eval_data)}')

    use_all_ref_tokens = cfg.data.use_all_ref_tokens
    max_length = cfg.data.max_length
    all_train_classifier_data = create_classifier_data(all_train_data, use_all_ref_tokens, max_length)

    all_train_length = len(all_train_classifier_data['input_ids'])
    id_eval_ratio = 0.1
    shuffled_indices = np.random.choice(all_train_length, all_train_length, replace=False)
    train_indices = shuffled_indices[:int(all_train_length * (1 - id_eval_ratio))]
    id_eval_indices = shuffled_indices[int(all_train_length * (1 - id_eval_ratio)):]

    train_classifier_data = {k: [all_train_classifier_data[k][i] for i in train_indices] for k in all_train_classifier_data}
    id_eval_classifier_data = {k: [all_train_classifier_data[k][i] for i in id_eval_indices] for k in all_train_classifier_data}

    eval_max_size = cfg.training.eval_max_size
    if eval_max_size != -1 and eval_max_size < len(id_eval_classifier_data['input_ids']):
        eval_random_indices = np.random.choice(len(id_eval_classifier_data['input_ids']), eval_max_size, replace=False)
        id_eval_classifier_data = {k: [id_eval_classifier_data[k][i] for i in eval_random_indices] for k in id_eval_classifier_data}

    print(f'Training samples: {len(train_classifier_data["input_ids"])}, ID eval samples: {len(id_eval_classifier_data["input_ids"])}')
    return train_classifier_data, id_eval_classifier_data
