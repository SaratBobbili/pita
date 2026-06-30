import json
from datasets.base import Example, GenerationTask


class PreferenceDataset:
    def __init__(self, cfg):
        self.name = cfg.dataset.name
        self.data_path = cfg.dataset.data_path

    def load(self, cfg) -> list[Example]:
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)

        examples = []
        for prompt_text, row in raw_data.items():
            examples.append(Example(
                example_id=row['id'],
                prompt=prompt_text,
                split=row['split'],
                extra={
                    'problem': prompt_text,
                    'output_1': row['output_1'],
                    'output_2': row['output_2'],
                    'preference': row['preference'],
                }
            ))
        examples.sort(key=lambda x: x.example_id)
        return examples

    def build_tasks(self, examples, num_repeats, num_context, base_seed) -> list[GenerationTask]:
        tasks = []
        for ex in examples:
            if ex.split != 'train':
                continue
            for r in range(num_repeats):
                for c in range(num_context):
                    seed = base_seed + 50 * r + 10 * c
                    tasks.append(GenerationTask(
                        example_id=ex.example_id,
                        repeat_id=r,
                        context_id=c,
                        seed=seed,
                    ))
        return tasks

    def generation_strategy(self) -> str:
        return "offline_pairs"
