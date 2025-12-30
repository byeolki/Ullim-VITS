import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from ullim_vits.utils.audio import load_wav, MelSpectrogram
from ullim_vits.utils.phonemizer import KoreanPhonemizer
from ullim_vits.data.text_processing import normalize_text


class VITSDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, config, split="train"):
        self.audio_dir = Path(audio_dir)
        self.config = config
        self.split = split
        self.segment_size = config.segment_size
        self.sampling_rate = config.sampling_rate

        self.phonemizer = KoreanPhonemizer()
        self.mel_transform = MelSpectrogram(
            n_fft=config.filter_length,
            hop_length=config.hop_length,
            win_length=config.win_length,
            sampling_rate=config.sampling_rate,
            n_mel_channels=config.n_mel_channels,
            mel_fmin=config.mel_fmin,
            mel_fmax=config.mel_fmax
        )

        self.data = []
        self.speaker_map = {}

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) != 3:
                    continue

                audio_name, speaker_id, text = parts

                if speaker_id not in self.speaker_map:
                    self.speaker_map[speaker_id] = len(self.speaker_map)

                self.data.append({
                    "audio_path": self.audio_dir / audio_name,
                    "speaker_id": self.speaker_map[speaker_id],
                    "text": text
                })

        print(f"Loaded {len(self.data)} samples from {split} split")
        print(f"Number of speakers: {len(self.speaker_map)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        audio = load_wav(item["audio_path"], self.sampling_rate)

        if self.split == "train" and audio.size(0) >= self.segment_size:
            max_start = audio.size(0) - self.segment_size
            start = random.randint(0, max_start)
            audio = audio[start:start + self.segment_size]

        mel = self.mel_transform(audio.unsqueeze(0)).squeeze(0)

        text = normalize_text(item["text"], self.config.text.cleaners)
        phonemes = self.phonemizer.text_to_sequence(text)
        phonemes = torch.LongTensor(phonemes)

        speaker_id = torch.LongTensor([item["speaker_id"]])

        return {
            "audio": audio,
            "mel": mel,
            "phonemes": phonemes,
            "speaker_id": speaker_id
        }

    @property
    def n_speakers(self):
        return len(self.speaker_map)
