import json
from training.dataset import read_jsonl
from datasets.base import Example, GenerationTask


class ArithmeticDataset:
    def __init__(self, cfg):
        self.name = cfg.dataset.name
        self.answer_key = cfg.dataset.answer_key
        self.data_path = cfg.dataset.data_path
        self.train_eval_save_path = cfg.dataset.train_eval_save_path

    def load(self, cfg) -> list[Example]:
        raw_data = read_jsonl(self.data_path)
        with open(self.train_eval_save_path, 'r') as f:
            train_eval_d = json.load(f)

        examples = []
        for i, row in enumerate(raw_data):
            problem = row['problem']
            if problem in train_eval_d:
                split = train_eval_d[problem]['split']
                eid = train_eval_d[problem]['id']
            else:
                split = 'train'
                eid = i
            examples.append(Example(
                example_id=eid,
                prompt=problem,
                split=split,
                extra={self.answer_key: str(row[self.answer_key]), 'problem': problem}
            ))
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
        return "guided"
