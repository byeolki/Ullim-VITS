import hydra
from omegaconf import DictConfig
import torch
import argparse
from pathlib import Path

from ullim_vits.inference.synthesizer import Synthesizer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--speaker_id", type=int, default=0)
    args = parser.parse_args()

    synthesizer = Synthesizer(config, args.checkpoint)

    audio = synthesizer.synthesize(args.text, speaker_id=args.speaker_id)

    synthesizer.save_audio(audio, args.output)

    print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
