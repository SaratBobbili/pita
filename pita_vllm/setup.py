# Adapted from SPPO / alignment-handbook setup.py for pita_vllm train deps.

import re
import shutil
from pathlib import Path

from setuptools import find_packages, setup


stale_egg_info = Path(__file__).parent / "pita_vllm.egg-info"
if stale_egg_info.exists():
    shutil.rmtree(stale_egg_info)

_deps = [
    "accelerate==0.27.2",
    "bitsandbytes==0.41.2.post2",
    "black==23.1.0",
    "datasets==2.14.6",
    "deepspeed==0.12.2",
    "einops>=0.6.1",
    "evaluate==0.4.0",
    "flake8>=6.0.0",
    "hf-doc-builder>=0.4.0",
    "hf_transfer>=0.1.4",
    "huggingface-hub>=0.19.2,<1.0",
    "hydra-core>=1.3",
    "isort>=5.12.0",
    "ninja>=1.11.1",
    "numpy==1.26.4",
    "omegaconf>=2.3",
    "packaging>=23.0",
    "parameterized>=0.9.0",
    "peft==0.7.1",
    "protobuf<=3.20.2",
    "pytest",
    "safetensors>=0.3.3",
    "sentencepiece>=0.1.99",
    "scipy",
    "tensorboard",
    "torch==2.1.2",
    "transformers==4.42.4",
    "trl==0.9.6",
    "jinja2>=3.0.0",
    "tqdm>=4.64.1",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ \[\]]+)(?:\[[^\]]+\])?(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["tests"] = deps_list("pytest", "parameterized")
extras["torch"] = deps_list("torch")
extras["quality"] = deps_list("black", "isort", "flake8")
extras["docs"] = deps_list("hf-doc-builder")
extras["dev"] = extras["docs"] + extras["quality"] + extras["tests"]

install_requires = [
    deps["accelerate"],
    deps["bitsandbytes"],
    deps["einops"],
    deps["evaluate"],
    deps["datasets"],
    deps["deepspeed"],
    deps["hf_transfer"],
    deps["huggingface-hub"],
    deps["hydra-core"],
    deps["jinja2"],
    deps["ninja"],
    deps["numpy"],
    deps["omegaconf"],
    deps["packaging"],
    deps["peft"],
    deps["protobuf"],
    deps["safetensors"],
    deps["sentencepiece"],
    deps["scipy"],
    deps["tensorboard"],
    deps["tqdm"],
    deps["transformers"],
    deps["trl"],
]

setup(
    name="pita-vllm",
    version="0.1.0.dev0",
    author="pita",
    description="PITA training (vLLM-guided preference / arithmetic)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="llm preference-optimization pita vllm",
    license="Apache",
    package_dir={"": "train"},
    packages=find_packages("train"),
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.10.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
