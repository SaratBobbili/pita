from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Example:
    example_id: int
    prompt: str
    split: str
    extra: dict = field(default_factory=dict)


@dataclass
class GenerationTask:
    example_id: int
    repeat_id: int
    context_id: int
    seed: int


class DatasetSpec(Protocol):
    name: str

    def load(self, cfg) -> list[Example]: ...
    def build_tasks(self, examples, num_repeats, num_context, base_seed) -> list[GenerationTask]: ...
    def generation_strategy(self) -> str: ...
