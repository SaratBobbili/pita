import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    from training.trainer import run_training
    run_training(cfg)


if __name__ == "__main__":
    main()
