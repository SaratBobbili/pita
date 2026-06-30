import ray
import torch
import json
import os
import math
import copy
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, set_seed

from generation.collector import load_models, get_match_fn, get_generate_kwargs
from generation.strategies import GuidedGenerate, GuidedPairwise, OfflinePairs


@ray.remote(num_gpus=1)
class GenerationWorker:
    def __init__(self, cfg):
        self.cfg = OmegaConf.create(cfg)
        self.device = torch.device("cuda")
        torch.set_grad_enabled(False)

    def _build_strategy(self):
        cfg = self.cfg
        strategy_name = cfg.dataset.generation_strategy

        if strategy_name == "offline_pairs":
            tokenizer = AutoTokenizer.from_pretrained(cfg.models.ref_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            return OfflinePairs(tokenizer, cfg)

        ref_model, classifier_model, tokenizer, logit_processor, logit_processor_disabled = load_models(cfg, self.device)
        match_fn = get_match_fn(cfg)
        generate_kwargs = get_generate_kwargs(cfg)

        if strategy_name == "guided":
            reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
            reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
            reward_tokenizer = AutoTokenizer.from_pretrained(reward_name)
            reward_model.eval()
            return GuidedGenerate(ref_model, classifier_model, tokenizer, logit_processor, logit_processor_disabled,
                                  reward_model, reward_tokenizer, cfg, match_fn, generate_kwargs)

        elif strategy_name == "guided_pairwise":
            scoring_model_id = "nvidia/OpenMath2-Llama3.1-8B"
            scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model_id)
            scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_id)
            scoring_model.eval()
            return GuidedPairwise(ref_model, classifier_model, tokenizer, logit_processor, logit_processor_disabled,
                                  scoring_model, scoring_tokenizer, cfg, match_fn, generate_kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def run(self, tasks, examples_dict, shard_path):
        strategy = self._build_strategy()
        cfg = self.cfg
        batch_size = cfg.generation.batch_size
        results = []

        task_groups = {}
        for t in tasks:
            key = (t.repeat_id, t.context_id)
            task_groups.setdefault(key, []).append(t)

        for (repeat_id, context_id), group_tasks in tqdm(task_groups.items(), desc="groups"):
            num_batches = math.ceil(len(group_tasks) / batch_size)
            for j in range(num_batches):
                batch_tasks = group_tasks[j * batch_size: (j + 1) * batch_size]
                batch_data = []
                for t in batch_tasks:
                    ex = examples_dict[t.example_id]
                    row = {'prompt': ex.prompt, 'id': ex.example_id, **ex.extra}
                    batch_data.append(row)

                seed = batch_tasks[0].seed
                batch_data = strategy.collect(batch_data, repeat_id, context_id, seed)

                for k, row in enumerate(batch_data):
                    row['repeat_id'] = repeat_id
                    row['context_id'] = context_id
                    results.append(row)

        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        df.to_parquet(shard_path)
        return shard_path
