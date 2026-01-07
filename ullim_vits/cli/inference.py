import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

from ullim_vits.inference.synthesizer import Synthesizer


@hydra.main(version_base=None, config_path="../../configs", config_name="inference")
def main(config: DictConfig):
    synthesizer = Synthesizer(config, config.checkpoint)

    audio = synthesizer.synthesize(
        config.text,
        speaker_id=config.get("speaker_id", 0),
        noise_scale=config.get("noise_scale", 0.667),
        length_scale=config.get("length_scale", 1.0)
    )

    synthesizer.save_audio(audio, config.output)

    print(f"Audio saved to {config.output}")


if __name__ == "__main__":
    main()
