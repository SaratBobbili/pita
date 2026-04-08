import glob
import json
import os
from tqdm import tqdm
from utils import read_jsonl, write_jsonl
import argparse
import datasets

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_template_path', default=None, type=str, help='')
parser.add_argument('--output_dir', default=None, type=str, help='directory where the jsonl file will be saved')
# parser.add_argument('--data_path', default=None, type=str, help='path to the data dataset/gsm8k_train.jsonl')
# parser.add_argument('--train_eval_save_path', default=None, type=str,
#                     help='train eval split dataset/gsm8k_train_eval.json')
args = parser.parse_args()
args_dict = vars(args)
# print(args_dict)

data_template_path = args.data_template_path
data_template_paths = [os.path.join(data_template_path, '*.json')]
output_path = os.path.join(args.output_dir, 'all_train_pref_data.jsonl')
additional_keys = ['fully_guided_predictions', 'fully_guided_predictions_correctness', 'partial_guided_prompts', 'partial_guided_prompts_tokenized', 'num_response_tokens_in_partial_guided_prompts', 'partial_guided_responses_tokenized', 'partial_guided_predictions', 'partial_guided_predictions_correctness']
# problem_data_path = args.data_path
# train_eval_data_path = args.train_eval_save_path
print('data_template_paths', data_template_paths)
print('output_path', output_path)
# problem_data = read_jsonl(problem_data_path)

# with open(train_eval_data_path, 'r') as f:
#     train_eval_data_d = json.load(f)

# Unlike with math reasoning, here we are just compiling of list of json objects into a single jsonl file
with open(output_path, "w") as out_f:
    for data_template_path in data_template_paths:
        for path in tqdm(glob.glob(data_template_path)):
            with open(path, 'r') as f:
                data = json.load(f)
                out_f.write(json.dumps(data) + "\n")
                # problem_index = train_eval_data_d[data['problem']]['id']
                # assert len(data['fully_guided_predictions']) == 1
                # problem_data[problem_index][key] = [data[key][0]]
                # for key in additional_keys:
                #     if key not in problem_data[problem_index]:
                #         problem_data[problem_index][key] = []
                #     assert len(data[key]) == 1
                #     problem_data[problem_index][key].append(data[key][0])

# write_jsonl(problem_data, output_path)
print('done')
