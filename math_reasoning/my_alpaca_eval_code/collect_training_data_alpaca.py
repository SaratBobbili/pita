import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, DataCollatorForLanguageModeling
from classifier import CustomLlamaForSequenceClassification, CustomValueGuidedLogitProcessor
from utils import read_jsonl, tokenize_with_chat_template, generate_with_classifier_guidance, get_parent_directory, resolve_dict_value, get_output_indices
import json
import os
import math
from tqdm import tqdm
import glob
import copy

parser = argparse.ArgumentParser(description='')
parser.add_argument('--start_index', default=0, type=int, help='start index for data')
parser.add_argument('--end_index', default=-1, type=int, help='end index for data, -1 means all data')

parser.add_argument('--data_path', default=None, type=str, help='path to the data (as a json array)')
# NOTE: defaulting to meta-llama/Meta-Llama-3-8B-Instruct since that is the only model we are using in this
# experiment.
parser.add_argument('--ref_model_id', default='meta-llama/Meta-Llama-3-8B-Instruct', type=str,
                    help='reference model id')
parser.add_argument('--classifier_model_id', default='meta-llama/Llama-3.2-1B-Instruct', type=str, help='classifier model id (for tokenizer, reuse weights)')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--num_samples', default=1, type=int, help='number of samples per problem')

parser.add_argument('--num_samples_context', default=1, type=int, help='number of comparison pairs per context per problem')

parser.add_argument('--use_chat_template', default=1, type=int, help='whether to use chat template for generation')

parser.add_argument('--max_new_tokens', default=None, type=int, help='max tokens for sampling 1024')
parser.add_argument('--dtype', default=None, type=str, help='data type for the model bfloat16')

parser.add_argument('--output_dir', default=None, type=str)

parser.add_argument('--seed', default=47, type=int, help='seed for reproduction')

parser.add_argument('--force', default=0, type=int, help='force overwrite existing files')

args = parser.parse_args()
args_dict = vars(args)
# print(args_dict)

start_index = args.start_index
end_index = args.end_index
force = args.force

seed = args.seed
ref_model_id = args.ref_model_id
classifier_model_id = args.classifier_model_id

data_path = args.data_path


batch_size = args.batch_size
num_samples = args.num_samples
num_samples_context = args.num_samples_context
use_chat_template = args.use_chat_template

dtype = args.dtype

output_dir = args.output_dir


os.makedirs(output_dir, exist_ok=True)

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ref_model_id)
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
assert len(tokenizer) == len(classifier_tokenizer), "tokenizer vocab size mismatch"
vocab_size = len(tokenizer)
if tokenizer.pad_token is None:
    assert 'Llama-3' in ref_model_id
    tokenizer.pad_token = tokenizer.added_tokens_decoder[128002].content  # reserved special token 0
tokenizer.padding_side = "left"  # for inference
print('tokenizer padding side:', tokenizer.padding_side)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)



with open(data_path, "r") as f:
    train_eval_problems_d = json.load(f)

train_data = []
eval_data = []
for instruction, row in train_eval_problems_d.items():
    sample = {
        "problem": instruction,
        "prompt": instruction,
        "id": row["id"],
        "split": row["split"],
        "output_1": row["output_1"],
        "output_2": row["output_2"],
        "preference": row["preference"],
    }
    if row["split"] == "train":
        train_data.append(sample)
    elif row["split"] == "eval":
        eval_data.append(sample)

train_data.sort(key=lambda x: x["id"])
eval_data.sort(key=lambda x: x["id"])

if end_index == -1:
    end_index = len(train_data)

print(f"loaded train={len(train_data)} eval={len(eval_data)} from {data_path}")


for i in range(num_samples):
    repeat_index = i
    print('repeat {0}'.format(repeat_index))
    for comparison_index in tqdm(range(num_samples_context)):
        if not force:
            existing_data_paths = glob.glob(os.path.join(output_dir, '*_c{0}_r{1}_alpaca.json'.format(comparison_index,repeat_index)))
            existing_indices = [int(os.path.basename(path).split('_')[0]) for path in existing_data_paths]
        else:
            existing_indices = []
        train_data_to_infer = []
        for j in range(start_index, end_index):
            if j not in existing_indices:
                train_data_to_infer.append(copy.deepcopy(train_data[j]))
        print('total number of problems to infer for repeat {0}:'.format(repeat_index), len(train_data_to_infer))
        num_batches = math.ceil(len(train_data_to_infer) / batch_size)
        for j in tqdm(range(num_batches)):
            batch_start_index = j * batch_size
            batch_end_index = min((j + 1) * batch_size, len(train_data_to_infer))
            batch_indices = list(range(batch_start_index, batch_end_index))
            current_prompts = [train_data_to_infer[k]['prompt'] for k in range(batch_start_index, batch_end_index)]

            current_inputs, current_formatted_prompts = tokenize_with_chat_template(tokenizer, current_prompts, use_chat_template, device)
            #current_outputs = generate_with_classifier_guidance(ref_model, tokenizer, logit_processor, current_inputs, generate_kwargs, True, False)
            current_outputs_text = [train_data_to_infer[k]["output_1"] for k in range(batch_start_index, batch_end_index)]
            current_outputs = tokenizer(
                current_outputs_text,
                add_special_tokens=False,
            )["input_ids"]

            current_batch_predictions_correctness = []
            for k in range(len(current_outputs_text)):
                prediction_correctness = train_data_to_infer[batch_indices[k]]['preference'] == 1
                current_batch_predictions_correctness.append(prediction_correctness)

            for k in range(len(current_outputs_text)):
                if 'fully_guided_predictions' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['fully_guided_predictions'] = []
                train_data_to_infer[batch_indices[k]]['fully_guided_predictions'].append(current_outputs_text[k])
                if 'fully_guided_predictions_correctness' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['fully_guided_predictions_correctness'] = []
                train_data_to_infer[batch_indices[k]]['fully_guided_predictions_correctness'].append(current_batch_predictions_correctness[k])

            # start randomly cutting responses
            # outputs_end_indices = (current_outputs == tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
            # outputs_end_indices = get_output_indices(current_outputs, tokenizer.eos_token_id)
            # outputs_lengths = outputs_end_indices + 1
            # random_cut_locations = torch.floor(torch.rand(outputs_lengths.size()).to(outputs_lengths.device) * outputs_lengths).int()
            # skip_inference_flags = [random_cut_locations[k] + 1 == outputs_lengths[k] for k in range(len(outputs_lengths))]
            # skip_inference_indices = [k for k in range(len(outputs_lengths)) if skip_inference_flags[k]]

            # queries is a list of unpadded input_ids
            queries = [current_inputs['input_ids'][k].masked_select(current_inputs['attention_mask'][k].to(torch.bool)) for k in range(len(current_inputs['input_ids']))]
            partial_responses = [train_data_to_infer[k]['output_2'] for k in range(batch_start_index, batch_end_index)]
            partial_guided_responses_tokenized = tokenizer(
                partial_responses,
                add_special_tokens=False,
            )["input_ids"]
            


            partial_guided_predictions = partial_responses

            current_batch_partial_guided_pred_correctness = []
            for k in range(len(queries)):
                prediction_preference = train_data_to_infer[batch_indices[k]]['preference'] == 2
                current_batch_partial_guided_pred_correctness.append(prediction_preference)

            for k in range(len(queries)):
                if 'fully_guided_responses_tokenized' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['fully_guided_responses_tokenized'] = []
                train_data_to_infer[batch_indices[k]]['fully_guided_responses_tokenized'].append(current_outputs[k])
                if 'partial_guided_prompts_tokenized' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['partial_guided_prompts_tokenized'] = []
                train_data_to_infer[batch_indices[k]]['partial_guided_prompts_tokenized'].append(current_inputs['input_ids'][k].tolist())
                if 'partial_guided_prompts' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['partial_guided_prompts'] = []
                train_data_to_infer[batch_indices[k]]['partial_guided_prompts'].append(tokenizer.decode(current_inputs['input_ids'][k].tolist()))
                if 'partial_guided_responses_tokenized' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['partial_guided_responses_tokenized'] = []
                train_data_to_infer[batch_indices[k]]['partial_guided_responses_tokenized'].append(partial_guided_responses_tokenized[k])
                if 'partial_guided_predictions' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['partial_guided_predictions'] = []
                train_data_to_infer[batch_indices[k]]['partial_guided_predictions'].append(partial_guided_predictions[k])
                # sanity check

                if 'partial_guided_predictions_correctness' not in train_data_to_infer[batch_indices[k]]:
                    train_data_to_infer[batch_indices[k]]['partial_guided_predictions_correctness'] = []
                train_data_to_infer[batch_indices[k]]['partial_guided_predictions_correctness'].append(current_batch_partial_guided_pred_correctness[k])
            # save individual problems in a batch
            for k in range(len(queries)):
                problem_id = train_data_to_infer[batch_indices[k]]["id"]
                current_problem_output_path = os.path.join(output_dir, f'{problem_id}_c{comparison_index}_r{repeat_index}.json')
                assert not os.path.exists(current_problem_output_path), f"problem {problem_id} data already exists for repeat {repeat_index}"
                with open(current_problem_output_path, 'w') as f:
                    json.dump(train_data_to_infer[batch_indices[k]], f, indent=4)
print('done')
