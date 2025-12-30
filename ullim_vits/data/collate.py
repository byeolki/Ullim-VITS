import torch
import torch.nn.functional as F


class VITSCollate:
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        max_audio_len = max([x["audio"].size(0) for x in batch])
        max_mel_len = max([x["mel"].size(1) for x in batch])
        max_phoneme_len = max([x["phonemes"].size(0) for x in batch])

        audio_padded = []
        mel_padded = []
        phoneme_padded = []
        speaker_ids = []
        audio_lengths = []
        mel_lengths = []
        phoneme_lengths = []

        for x in batch:
            audio = x["audio"]
            mel = x["mel"]
            phonemes = x["phonemes"]

            audio_lengths.append(audio.size(0))
            mel_lengths.append(mel.size(1))
            phoneme_lengths.append(phonemes.size(0))

            audio_padded.append(F.pad(audio, (0, max_audio_len - audio.size(0))))
            mel_padded.append(F.pad(mel, (0, max_mel_len - mel.size(1))))
            phoneme_padded.append(F.pad(phonemes, (0, max_phoneme_len - phonemes.size(0))))
            speaker_ids.append(x["speaker_id"])

        audio_padded = torch.stack(audio_padded)
        mel_padded = torch.stack(mel_padded)
        phoneme_padded = torch.stack(phoneme_padded)
        speaker_ids = torch.stack(speaker_ids).squeeze(1)
        audio_lengths = torch.LongTensor(audio_lengths)
        mel_lengths = torch.LongTensor(mel_lengths)
        phoneme_lengths = torch.LongTensor(phoneme_lengths)

        return {
            "audio": audio_padded,
            "mel": mel_padded,
            "phonemes": phoneme_padded,
            "speaker_id": speaker_ids,
            "audio_lengths": audio_lengths,
            "mel_lengths": mel_lengths,
            "phoneme_lengths": phoneme_lengths
        }
