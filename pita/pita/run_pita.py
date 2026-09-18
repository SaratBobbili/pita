"""Training entry point.

Derived from ``SPPO/sppo/run_sppo.py``, with its dead paths dropped: the unused
``load_config`` helper, the vestigial in-process ``num_iteration`` loop (rounds are driven
by ``run_pita_<model>.sh``), the force-disabled eval branch, the resume-from-checkpoint
path that always passed ``None``, and the PEFT wiring that the trainer raised on.

Usage mirrors SPPO's::

    accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml \\
        -m pita.run_pita recipes/pita/llama3.yaml --output_dir=... --train_file=...
"""

import logging
import sys

import transformers

from pita.configs import DataArguments, H4ArgumentParser, ModelArguments, PITAConfig
from pita.trainer import train

logger = logging.getLogger(__name__)


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, PITAConfig))
    model_args, data_args, training_args = parser.parse()
    setup_logging()

    logger.info(f"Reference policy (frozen): {model_args.ref_model_id}")
    logger.info(f"Classifier backbone: {model_args.classifier_model_id}")
    logger.info(f"Resuming classifier from: {model_args.classifier_path or '(fresh head)'}")
    logger.info(f"Training data: {data_args.train_file}")

    metrics = train(model_args, data_args, training_args)
    logger.info(f"Done. {metrics}")


if __name__ == "__main__":
    main()
