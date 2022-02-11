import os

import dotenv
import hydra
from omegaconf import DictConfig

from emma_policy import train_model


dotenv.load_dotenv(override=True)


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    """Run the model."""
    train_model(config)


if __name__ == "__main__":
    main()
