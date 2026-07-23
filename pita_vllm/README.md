# pita_vllm

Product tree for PITA: `train/` (rounds of generate + classifier train) and `evaluation/` (final eval, deferred).

## Layout

```text
pita_vllm/
  train/
    dataset/                 # dataset configs (preference, math, …)
    recipes/{sppo,ipo,kto,pita}/
    utils/                   # shared helpers
    configs/                 # shared infra (accelerate, …)
    launch.sh                # entry stub
  evaluation/                # stub; hierarchy later
  setup.py
```

PITA family model pairs: `train/recipes/pita/configs/{llama,qwen,mistral}.yaml`.

## Environment setup

Do **not** run these installs unless you intend to create the env (S0 delivers files + docs only).

Baseline follows SPPO (conda 3.10, vLLM, PairRM/LLM-Blender, editable install). PITA extras (`hydra-core`, `omegaconf`) are included in `setup.py`.

1. **Create env**

   ```bash
   conda create -n pita python=3.10
   conda activate pita
   ```

2. **Install vLLM** (generation)

   ```bash
   pip install vllm
   ```

3. **Install PairRM** (SPPO ranking baseline; optional until ranking is wired)

   ```bash
   git clone https://github.com/yuchenlin/LLM-Blender.git
   cd LLM-Blender
   pip install -e .
   cd ..
   ```

4. **Install train deps** (from this directory)

   ```bash
   cd pita_vllm
   pip install -e .
   ```

### SPPO baseline vs PITA extras

| Source | Packages / steps |
|--------|------------------|
| SPPO | `vllm`, LLM-Blender, Accelerate/DeepSpeed/TRL stack in `setup.py` |
| PITA (`refactor_old`) | `hydra-core`, `omegaconf` (already in `setup.py`) |

Torch / CUDA pins match SPPO (`torch==2.1.2`). Revisit if a newer vLLM requires different pins.

## Launch

Shared Accelerate configs live under `train/configs/accelerate_configs/`. Use `train/launch.sh` as the entry stub once recipe scripts exist.
