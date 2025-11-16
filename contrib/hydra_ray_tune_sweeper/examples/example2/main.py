import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    result = cfg.x + cfg.y
    print(result)
    return result


if __name__ == "__main__":
    main()
