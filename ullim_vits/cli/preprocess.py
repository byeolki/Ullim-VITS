import hydra
from omegaconf import DictConfig
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.download_zeroth import download_zeroth
from tools.preprocess_dataset import preprocess_dataset


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig):
    data_dir = config.data.data_dir

    print("Downloading Zeroth-Korean dataset...")
    download_zeroth(output_dir=data_dir, sample_rate=config.data.sampling_rate)

    print("Preprocessing dataset...")
    preprocess_dataset(data_dir=data_dir)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
