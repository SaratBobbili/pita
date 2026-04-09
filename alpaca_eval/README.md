# AlpacaEval experiments (helpful assistant)

## Requirements
For training models and generating responses, the same environment used for math reasoning will work here. For running AlpacaEval, you will need an environment with vllm 0.18.0 and alpaca_eval 0.6.6 installed.

## Warnings
AlpacaEval will cache comparison results in a json file within the evaluator_configs directory under the directory for the annotator which is being used (in this case, weighted_alpaca_eval_vllm_llama3_70b). The name of the file will be something like `annotations_seed0_configs.json`. If the cache file already exists and you try to do a comparison, it might reuse the cached comparisons instead of starting from scratch even if the json file you are running the evaluation on is different. The eval_json_vllm.sh script will attempt to check for the cached file and warn you about it.








