import os
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf

from datasets.arithmetic import ArithmeticDataset
from datasets.preference import PreferenceDataset
from generation.worker import GenerationWorker


def get_dataset_spec(cfg):
    if cfg.dataset.family == "arithmetic":
        return ArithmeticDataset(cfg)
    elif cfg.dataset.family == "preference":
        return PreferenceDataset(cfg)
    else:
        raise ValueError(f"Unknown dataset family: {cfg.dataset.family}")


def read_existing_parquet_keys(shards_dir):
    done_keys = set()
    if not os.path.exists(shards_dir):
        return done_keys
    for f in os.listdir(shards_dir):
        if f.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(shards_dir, f))
            for _, row in df.iterrows():
                done_keys.add((row.get('id'), row.get('repeat_id'), row.get('context_id')))
    return done_keys


def merge_parquets(shard_paths, output_path):
    dfs = [pd.read_parquet(p) for p in shard_paths if p is not None]
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_parquet(output_path)
        print(f"Merged {len(dfs)} shards -> {output_path} ({len(merged)} rows)")


def run_generation(cfg):
    ray.init(num_gpus=cfg.trainer.n_gpus_per_node, ignore_reinit_error=True)

    dataset_spec = get_dataset_spec(cfg)
    examples = dataset_spec.load(cfg)

    num_repeats = cfg.generation.num_repeats
    num_context = cfg.generation.get('num_context', 1)
    base_seed = cfg.generation.seed
    tasks = dataset_spec.build_tasks(examples, num_repeats, num_context, base_seed)

    shards_dir = os.path.join(cfg.trainer.output_dir, 'shards')
    done_keys = read_existing_parquet_keys(shards_dir)
    if done_keys:
        tasks = [t for t in tasks if (t.example_id, t.repeat_id, t.context_id) not in done_keys]
        print(f"Resuming: {len(done_keys)} done, {len(tasks)} remaining")

    if not tasks:
        print("All tasks already completed.")
        return

    n_workers = cfg.trainer.n_gpus_per_node
    shards = np.array_split(tasks, n_workers)

    examples_dict = {ex.example_id: ex for ex in examples}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    workers = [GenerationWorker.remote(cfg_dict) for _ in range(n_workers)]
    futures = []
    for i, (w, s) in enumerate(zip(workers, shards)):
        shard_path = os.path.join(shards_dir, f'shard_{i}.parquet')
        futures.append(w.run.remote(list(s), examples_dict, shard_path))

    shard_paths = ray.get(futures)
    output_path = os.path.join(cfg.trainer.output_dir, 'train_data.parquet')
    merge_parquets(shard_paths, output_path)
    ray.shutdown()
