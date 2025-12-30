import os
import torch
import torchaudio
from datasets import load_dataset
from pathlib import Path
import argparse
from tqdm import tqdm


def download_zeroth(output_dir="data/zeroth", sample_rate=22050):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Zeroth-Korean dataset from HuggingFace...")
    dataset = load_dataset("kresnik/zeroth_korean")

    for split in ["train", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)

        audio_dir = split_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        metadata = []

        print(f"Processing {split} split...")
        for idx, item in enumerate(tqdm(dataset[split])):
            audio = item["audio"]
            text = item["text"]
            speaker_id = item.get("speaker_id", "unknown")

            audio_array = audio["array"]
            original_sr = audio["sampling_rate"]

            audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)

            if original_sr != sample_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, original_sr, sample_rate
                )

            audio_path = audio_dir / f"{speaker_id}_{idx}.wav"
            torchaudio.save(str(audio_path), audio_tensor, sample_rate)

            metadata.append(f"{audio_path.name}|{speaker_id}|{text}\n")

        metadata_path = split_dir / "metadata.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.writelines(metadata)

        print(f"{split} split: {len(metadata)} files saved")

    print("Download complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/zeroth")
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    download_zeroth(args.output_dir, args.sample_rate)
