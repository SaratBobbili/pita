"""Package metadata for PITA.

Versions are deliberately not pinned the way SPPO's setup.py pins them (torch==2.1.2,
transformers==4.42.4): those predate the vLLM V1 logits-processor API this project is
built on. The floors below are what the code actually requires.

Ranking needs ``llm-blender`` (PairRM) and judging needs ``alpaca-eval``; both are listed
as extras because ``scripts/rank.py`` and the AlpacaEval judge each run as their own
process, so neither has to share an environment with the trainer.
"""

from setuptools import find_packages, setup

install_requires = [
    "torch>=2.6",
    "transformers>=4.51",
    "accelerate>=1.0",
    "vllm>=0.10",          # V1 custom logits processors (LogitsProcessor/BatchUpdate)
    "datasets>=3.0",
    "safetensors>=0.4",
    "numpy>=1.26",
    "pandas>=2.0",
    "pyarrow>=14.0",       # parquet round-tripping between rounds
    "tqdm>=4.64",
    "pyyaml>=6.0",
]

extras = {
    "rank": ["llm-blender"],        # PairRM preference oracle
    "eval": ["alpaca-eval>=0.6"],   # AlpacaEval 2 judge
    "tests": ["pytest"],
    "logging": ["wandb"],
}
extras["dev"] = sorted({dep for group in extras.values() for dep in group})

setup(
    name="pita",
    version="0.1.0",
    description="PITA: preference-guided inference-time alignment",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    packages=find_packages(include=["pita", "pita.*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras,
)
