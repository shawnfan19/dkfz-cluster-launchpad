from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    run_name: None | str = "default"


def train(cfg: TrainConfig):
    pass


def main():
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    if hasattr(cli_cfg, "config"):
        file_cfg = OmegaConf.load(cli_cfg.config)
        del cli_cfg.config
        cfg = OmegaConf.merge(cfg, file_cfg)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_object(cfg)
    train(cfg) # type: ignore


if __name__=="__main__":
    main()

