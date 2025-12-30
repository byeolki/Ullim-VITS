import torch
import torch.nn.functional as F


def mel_spectrogram_loss(mel_real, mel_generated):
    return F.l1_loss(mel_real, mel_generated)


def multi_scale_mel_loss(mel_real, mel_generated, scales=[5, 10, 20]):
    loss = 0

    for scale in scales:
        if scale > 1:
            mel_real_downsampled = F.avg_pool1d(mel_real, kernel_size=scale, stride=scale)
            mel_generated_downsampled = F.avg_pool1d(mel_generated, kernel_size=scale, stride=scale)
        else:
            mel_real_downsampled = mel_real
            mel_generated_downsampled = mel_generated

        loss += F.l1_loss(mel_real_downsampled, mel_generated_downsampled)

    return loss / len(scales)
