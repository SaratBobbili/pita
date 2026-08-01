"""Reuse DPO prepared preference pairs (prompt/chosen/rejected)."""
import os
import shutil
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_MR = Path(__file__).resolve().parents[2]
if str(_MR) not in sys.path:
    sys.path.insert(0, str(_MR))


@hydra.main(version_base=None, config_path="configs", config_name="train_reward")
def main(cfg: DictConfig):
    root = Path(cfg.math_reasoning_root) if cfg.math_reasoning_root else _MR
    os.chdir(root)
    print(OmegaConf.to_yaml(cfg))

    src = root / "recipes" / "DPO" / "data"
    dst = root / "recipes" / "PPO_RLHF" / "data"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run recipes/DPO/prepare_data.py first."
        )
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    os.symlink(src, dst)
    print(f"Linked {dst} -> {src}")


if __name__ == "__main__":
    main()
