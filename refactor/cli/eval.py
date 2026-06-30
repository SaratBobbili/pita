import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    if cfg.dataset.eval_type == "arithmetic":
        from eval.arithmetic import run_eval
    else:
        from eval.preference import run_eval
    run_eval(cfg)


if __name__ == "__main__":
    main()
