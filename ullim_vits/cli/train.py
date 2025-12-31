import warnings
warnings.filterwarnings('ignore', message='Could not initialize NNPACK') # hate nnpack errors

import hydra
from omegaconf import DictConfig
from ullim_vits.training.trainer import Trainer

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig):
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
