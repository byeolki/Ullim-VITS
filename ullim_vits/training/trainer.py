import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from ullim_vits.data.collate import VITSCollate
from ullim_vits.data.dataset import VITSDataset
from ullim_vits.losses.adversarial import discriminator_loss, generator_loss
from ullim_vits.losses.feature_matching import feature_matching_loss
from ullim_vits.losses.kl_loss import duration_loss, kl_divergence_loss
from ullim_vits.losses.reconstruction import mel_spectrogram_loss
from ullim_vits.models.discriminator.mpd import MultiPeriodDiscriminator
from ullim_vits.models.discriminator.msd import MultiScaleDiscriminator
from ullim_vits.models.vits.vits import VITS
from ullim_vits.training.optimizer import get_optimizer
from ullim_vits.training.scheduler import get_scheduler
from ullim_vits.utils.audio import MelSpectrogram
from ullim_vits.utils.logging import Logger, load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = GradScaler(enabled=config.train.mixed_precision)

        self.mel_transform = MelSpectrogram(
            n_fft=config.data.filter_length,
            hop_length=config.data.hop_length,
            win_length=config.data.win_length,
            sampling_rate=config.data.sampling_rate,
            n_mel_channels=config.data.n_mel_channels,
            mel_fmax=config.data.mel_fmax,
        ).to(self.device)

        self.logger = Logger(config)

        self.step = 0
        self.epoch = 0

    def setup_dataloaders(self):
        train_dataset = VITSDataset(
            metadata_path=Path(self.config.data.data_dir) / "train" / "metadata.txt",
            audio_dir=Path(self.config.data.data_dir) / "train" / "audio",
            config=self.config.data,
            split="train",
        )

        val_dataset = VITSDataset(
            metadata_path=Path(self.config.data.data_dir) / "test" / "metadata.txt",
            audio_dir=Path(self.config.data.data_dir) / "test" / "audio",
            config=self.config.data,
            split="test",
        )

        self.config.model.n_vocab = train_dataset.phonemizer.vocab_size

        self.model = VITS(self.config).to(self.device)
        self.mpd = MultiPeriodDiscriminator(self.config.model.discriminator.mpd.periods).to(
            self.device
        )
        self.msd = MultiScaleDiscriminator(self.config.model.discriminator.msd.scales).to(
            self.device
        )

        self.optimizer_g, self.optimizer_d = get_optimizer(
            self.model, nn.ModuleList([self.mpd, self.msd]), self.config
        )
        self.scheduler_g = get_scheduler(self.optimizer_g, self.config)
        self.scheduler_d = get_scheduler(self.optimizer_d, self.config)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            collate_fn=VITSCollate(),
            pin_memory=self.config.train.pin_memory,
            persistent_workers=self.config.train.persistent_workers,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            collate_fn=VITSCollate(),
            pin_memory=self.config.train.pin_memory,
        )

    def train_step(self, batch):
        audio = batch["audio"].to(self.device)
        mel = batch["mel"].to(self.device)
        phonemes = batch["phonemes"].to(self.device)
        speaker_id = batch["speaker_id"].to(self.device)
        audio_lengths = batch["audio_lengths"].to(self.device)
        mel_lengths = batch["mel_lengths"].to(self.device)
        phoneme_lengths = batch["phoneme_lengths"].to(self.device)

        with autocast("cuda", enabled=self.config.train.mixed_precision):
            (
                y_hat,
                ids_slice,
                phoneme_mask,
                mel_mask,
                (z_p, z_q, m_p, logs_p, m_q, logs_q),
                (logw, logw_),
                g,
            ) = self.model(phonemes, phoneme_lengths, mel, mel_lengths, speaker_id)

            y = torch.zeros_like(y_hat)
            for i in range(audio.size(0)):
                start = ids_slice[i] * self.config.data.hop_length
                end = start + y_hat.size(-1)
                audio_slice = audio[i, start:end]
                slice_len = min(audio_slice.size(0), y_hat.size(-1))
                y[i, :, :slice_len] = audio_slice[:slice_len]

            y_mel = self.mel_transform(y.squeeze(1))
            y_hat_mel = self.mel_transform(y_hat.squeeze(1))

        self.optimizer_d.zero_grad()

        with autocast("cuda", enabled=self.config.train.mixed_precision):
            y_d_hat_r, y_d_hat_g, _, _ = self.mpd(y, y_hat.detach())
            loss_disc_mpd, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            y_d_hat_r, y_d_hat_g, _, _ = self.msd(y, y_hat.detach())
            loss_disc_msd, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            loss_disc = loss_disc_mpd + loss_disc_msd

            # Check for NaN/Inf in discriminator loss
            if not torch.isfinite(loss_disc):
                print(
                    f"Warning: Non-finite discriminator loss detected: {loss_disc.item()}, skipping discriminator update"
                )
                loss_disc = torch.tensor(0.0, device=self.device, requires_grad=True)

        if torch.isfinite(loss_disc) and loss_disc.item() > 0:
            self.scaler.scale(loss_disc).backward()
            self.scaler.unscale_(self.optimizer_d)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                list(self.mpd.parameters()) + list(self.msd.parameters()),
                self.config.train.gradient_clip_val,
            )
            if torch.isfinite(grad_norm_d):
                self.scaler.step(self.optimizer_d)

        self.optimizer_g.zero_grad()

        with autocast("cuda", enabled=self.config.train.mixed_precision):
            loss_mel = (
                mel_spectrogram_loss(y_mel, y_hat_mel) * self.config.train.losses.mel_loss_weight
            )

            segment_size = self.config.data.segment_size // self.config.data.hop_length
            mel_mask_slice = torch.ones(
                z_p.size(0), 1, segment_size, dtype=z_p.dtype, device=z_p.device
            )

            loss_kl = (
                kl_divergence_loss(z_q, m_q, logs_q, m_p, logs_p, mel_mask_slice)
                * self.config.train.losses.kl_loss_weight
            )

            loss_dur = (
                duration_loss(logw, logw_, phoneme_lengths)
                * self.config.train.losses.duration_loss_weight
            )

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.mpd(y, y_hat)

            loss_gen_mpd, _ = generator_loss(y_d_hat_g)
            loss_fm_mpd = feature_matching_loss(fmap_r, fmap_g)

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.msd(y, y_hat)

            loss_gen_msd, _ = generator_loss(y_d_hat_g)
            loss_fm_msd = feature_matching_loss(fmap_r, fmap_g)

            loss_gen = (loss_gen_mpd + loss_gen_msd) * self.config.train.losses.gen_loss_weight
            loss_fm = (loss_fm_mpd + loss_fm_msd) * self.config.train.losses.feature_loss_weight

            loss_g = loss_mel + loss_kl + loss_dur + loss_gen + loss_fm

            # Check for NaN/Inf in generator loss
            if not torch.isfinite(loss_g):
                print(
                    f"Warning: Non-finite generator loss detected: {loss_g.item()}, skipping generator update"
                )
                loss_g = loss_mel + loss_kl + loss_dur  # Use only stable losses

        if torch.isfinite(loss_g) and loss_g.item() > 0:
            self.scaler.scale(loss_g).backward()
            self.scaler.unscale_(self.optimizer_g)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.gradient_clip_val
            )
            if torch.isfinite(grad_norm_g):
                self.scaler.step(self.optimizer_g)

        self.scaler.update()

        return {
            "loss_g": loss_g.item() if torch.isfinite(loss_g) else 0.0,
            "loss_d": loss_disc.item() if torch.isfinite(loss_disc) else 0.0,
            "loss_mel": loss_mel.item() if torch.isfinite(loss_mel) else 0.0,
            "loss_kl": loss_kl.item() if torch.isfinite(loss_kl) else 0.0,
            "loss_dur": loss_dur.item() if torch.isfinite(loss_dur) else 0.0,
            "loss_gen": loss_gen.item() if torch.isfinite(loss_gen) else 0.0,
            "loss_fm": loss_fm.item() if torch.isfinite(loss_fm) else 0.0,
        }

    def validate(self):
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                audio = batch["audio"].to(self.device)
                mel = batch["mel"].to(self.device)
                phonemes = batch["phonemes"].to(self.device)
                speaker_id = batch["speaker_id"].to(self.device)
                mel_lengths = batch["mel_lengths"].to(self.device)
                phoneme_lengths = batch["phoneme_lengths"].to(self.device)

                with autocast("cuda", enabled=self.config.train.mixed_precision):
                    (
                        y_hat,
                        ids_slice,
                        phoneme_mask,
                        mel_mask,
                        (z_p, z_q, m_p, logs_p, m_q, logs_q),
                        (logw, logw_),
                        g,
                    ) = self.model(phonemes, phoneme_lengths, mel, mel_lengths, speaker_id)

                    y = torch.zeros_like(y_hat)
                    for i in range(audio.size(0)):
                        start = ids_slice[i] * self.config.data.hop_length
                        end = start + y_hat.size(-1)
                        audio_slice = audio[i, start:end]
                        slice_len = min(audio_slice.size(0), y_hat.size(-1))
                        y[i, :, :slice_len] = audio_slice[:slice_len]

                    y_mel = self.mel_transform(y.squeeze(1))
                    y_hat_mel = self.mel_transform(y_hat.squeeze(1))

                    loss_mel = mel_spectrogram_loss(y_mel, y_hat_mel)

                    total_loss += loss_mel.item()

        avg_loss = total_loss / len(self.val_loader)

        self.model.train()

        return avg_loss

    def train(self):
        self.setup_dataloaders()

        self.model.train()
        self.mpd.train()
        self.msd.train()

        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.train.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                losses = self.train_step(batch)

                if self.step % self.config.train.log_every == 0:
                    self.logger.log(losses, self.step)
                    print(f"Epoch {epoch}, Step {self.step}: {losses}")

                if self.step % self.config.train.eval_every == 0:
                    val_loss = self.validate()
                    self.logger.log({"val_loss": val_loss}, self.step)
                    print(f"Validation loss: {val_loss}")

                if self.step % self.config.train.save_every == 0:
                    save_checkpoint(
                        {
                            "step": self.step,
                            "epoch": epoch,
                            "model": self.model.state_dict(),
                            "mpd": self.mpd.state_dict(),
                            "msd": self.msd.state_dict(),
                            "optimizer_g": self.optimizer_g.state_dict(),
                            "optimizer_d": self.optimizer_d.state_dict(),
                        },
                        checkpoint_dir,
                        self.step,
                    )

                self.step += 1

            if self.scheduler_g is not None:
                self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

        self.logger.finish()
