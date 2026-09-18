# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/huggingface/alignment-handbook
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


@dataclass
class ModelArguments:
    """The frozen reference policy and the value classifier trained against it."""

    ref_model_id: str = field(
        metadata={"help": "Frozen policy whose logits get tilted. Never trained."}
    )
    classifier_model_id: str = field(
        metadata={"help": "Backbone the value classifier is initialised from. Must share "
                          "a tokenizer with ref_model_id."}
    )
    classifier_path: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint from the previous round. None starts from "
                          "classifier_model_id with a fresh head."},
    )
    head_type: str = field(default="Q", metadata={"help": "Q (per-token values) or V (state value)."})
    loss_type: str = field(default="bce", metadata={"help": "bce, mse, or mle."})
    use_bias: bool = field(default=False, metadata={"help": "Bias term on the value head."})
    init_mode: str = field(
        default="reuse",
        metadata={"help": "reuse (warm-start a Q head from the backbone LM head), zero, or none."},
    )
    num_atoms: int = field(default=11, metadata={"help": "Atoms for the mle head."})
    V_min: float = field(default=0.0, metadata={"help": "Lowest atom value."})
    V_max: float = field(default=1.0, metadata={"help": "Highest atom value."})
    dtype: str = field(default="bfloat16", metadata={"help": "Torch dtype for the classifier."})


@dataclass
class DataArguments:
    train_file: str = field(
        metadata={"help": "Parquet from scripts/build_dataset.py: prompt, response, reward."}
    )
    max_length: int = field(default=-1, metadata={"help": "Truncate examples to this many tokens; -1 to keep all."})
    use_all_response_tokens: bool = field(
        default=True,
        metadata={"help": "Supervise every response token. False supervises only the first decision."},
    )
    eval_ratio: float = field(default=0.1, metadata={"help": "Fraction of prompts held out."})
    eval_max_size: int = field(default=1000, metadata={"help": "Cap on held-out examples; -1 for no cap."})
    preprocessing_num_workers: int = field(default=8, metadata={"help": "Dataloader workers."})


@dataclass
class PITAConfig:
    """Training loop settings.

    Deliberately not a transformers TrainingArguments: the loop is a plain Accelerate
    loop over a <=2B model, so the vast majority of those fields would be inert.
    """

    output_dir: str = field(metadata={"help": "Where round checkpoints are written."})
    batch_size: int = field(default=8, metadata={"help": "Per-device batch size."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Steps per optimizer update."})
    num_epochs: int = field(default=1, metadata={"help": "Passes over the round's data."})
    learning_rate: float = field(default=2e-5, metadata={"help": "AdamW learning rate."})
    weight_decay: float = field(default=0.01, metadata={"help": "AdamW weight decay."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup steps."})
    max_grad_norm: float = field(default=5.0, metadata={"help": "Gradient clipping norm."})
    eval_freq: int = field(default=500, metadata={"help": "Optimizer steps between evals; -1 disables."})
    ckpt_freq: int = field(default=-1, metadata={"help": "Optimizer steps between checkpoints; -1 saves only at the end."})
    seed: int = field(default=47, metadata={"help": "Base seed."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Trade compute for activation memory."})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Enables W&B when set."})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "W&B entity."})
    run_name: Optional[str] = field(default=None, metadata={"help": "W&B run name."})
