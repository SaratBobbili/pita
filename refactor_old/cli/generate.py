import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(cfg: DictConfig):
    from generation.driver import run_generation
    run_generation(cfg)


if __name__ == "__main__":
    main()
