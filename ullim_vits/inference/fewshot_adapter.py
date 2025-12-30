import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from ullim_vits.models.vits.vits import VITS
from ullim_vits.data.dataset import VITSDataset
from ullim_vits.data.collate import VITSCollate
from ullim_vits.utils.logging import load_checkpoint, save_checkpoint


class FewShotAdapter:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VITS(config).to(self.device)

        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.model.load_state_dict(checkpoint["model"])

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.speaker_encoder.parameters():
            param.requires_grad = True

        if hasattr(self.model, 'emb_g'):
            for param in self.model.emb_g.parameters():
                param.requires_grad = True

    def adapt(self, reference_audio_dir, reference_metadata, target_speaker_id, n_epochs=100, lr=1e-4):
        dataset = VITSDataset(
            metadata_path=reference_metadata,
            audio_dir=reference_audio_dir,
            config=self.config.data,
            split="train"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=VITSCollate()
        )

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

        self.model.train()

        for epoch in range(n_epochs):
            total_loss = 0

            for batch in dataloader:
                audio = batch["audio"].to(self.device)
                mel = batch["mel"].to(self.device)
                phonemes = batch["phonemes"].to(self.device)
                mel_lengths = batch["mel_lengths"].to(self.device)
                phoneme_lengths = batch["phoneme_lengths"].to(self.device)

                optimizer.zero_grad()

                speaker_id = torch.LongTensor([target_speaker_id]).to(self.device)

                y_hat, ids_slice, phoneme_mask, mel_mask, _, _, _ = self.model(
                    phonemes, phoneme_lengths, mel, mel_lengths, speaker_id
                )

                y = torch.zeros_like(y_hat)
                for i in range(audio.size(0)):
                    start = ids_slice[i] * self.config.data.hop_length
                    end = start + y_hat.size(-1)
                    y[i, :, :] = audio[i, start:end]

                loss = nn.functional.l1_loss(y_hat, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

        self.model.eval()

        return self.model

    def save_adapted_model(self, output_path):
        save_checkpoint({
            "model": self.model.state_dict(),
        }, Path(output_path).parent, "adapted")
