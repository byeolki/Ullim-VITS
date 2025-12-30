import argparse
from pathlib import Path
import torchaudio
import torch
from tqdm import tqdm
import json


def compute_statistics(audio_dir, metadata_path):
    mel_means = []
    mel_stds = []

    from ullim_vits.utils.audio import MelSpectrogram

    mel_transform = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Computing dataset statistics...")
    for line in tqdm(lines[:1000]):
        audio_name = line.split("|")[0]
        audio_path = audio_dir / audio_name

        if not audio_path.exists():
            continue

        audio, sr = torchaudio.load(audio_path)
        mel = mel_transform(audio)

        mel_means.append(mel.mean().item())
        mel_stds.append(mel.std().item())

    stats = {
        "mel_mean": sum(mel_means) / len(mel_means),
        "mel_std": sum(mel_stds) / len(mel_stds)
    }

    return stats


def preprocess_dataset(data_dir, output_stats_path="data/stats.json"):
    data_dir = Path(data_dir)

    for split in ["train", "test"]:
        split_dir = data_dir / split
        audio_dir = split_dir / "audio"
        metadata_path = split_dir / "metadata.txt"

        if not metadata_path.exists():
            print(f"Metadata not found: {metadata_path}")
            continue

        print(f"Processing {split} split...")

        if split == "train":
            stats = compute_statistics(audio_dir, metadata_path)
            with open(output_stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to {output_stats_path}")

    print("Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/zeroth")
    parser.add_argument("--output_stats", type=str, default="data/stats.json")
    args = parser.parse_args()

    preprocess_dataset(args.data_dir, args.output_stats)
