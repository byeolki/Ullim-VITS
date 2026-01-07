import math

import torch
import torch.nn as nn

from ullim_vits.models.speaker_encoder.encoder import SpeakerEncoder
from ullim_vits.models.vits.decoder import HiFiGANDecoder
from ullim_vits.models.vits.duration_predictor import StochasticDurationPredictor
from ullim_vits.models.vits.posterior_encoder import PosteriorEncoder
from ullim_vits.models.vits.prior_encoder import PriorEncoder
from ullim_vits.utils.alignment import generate_path, maximum_path


class VITS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_vocab = getattr(config.model, "n_vocab", 100)
        self.hidden_channels = config.model.hidden_channels
        self.n_speakers = config.model.n_speakers
        self.speaker_embed_dim = config.model.speaker_embed_dim

        self.speaker_encoder = SpeakerEncoder(
            input_dim=config.data.n_mel_channels,
            hidden_dim=self.speaker_embed_dim,
            output_dim=self.speaker_embed_dim,
        )

        if self.n_speakers > 1:
            self.emb_g = nn.Embedding(self.n_speakers, self.speaker_embed_dim)

        self.posterior_encoder = PosteriorEncoder(
            in_channels=config.data.n_mel_channels,
            out_channels=self.hidden_channels,
            hidden_channels=config.model.posterior_encoder.hidden_channels,
            kernel_size=config.model.posterior_encoder.kernel_size,
            dilation_rate=config.model.posterior_encoder.dilation_rate,
            n_layers=config.model.posterior_encoder.n_layers,
            gin_channels=self.speaker_embed_dim,
        )

        self.prior_encoder = PriorEncoder(
            n_vocab=self.n_vocab,
            out_channels=self.hidden_channels,
            hidden_channels=config.model.prior_encoder.hidden_channels,
            filter_channels=config.model.prior_encoder.filter_channels,
            n_heads=config.model.prior_encoder.n_heads,
            n_layers=config.model.prior_encoder.n_layers,
            kernel_size=config.model.prior_encoder.kernel_size,
            p_dropout=config.model.prior_encoder.p_dropout,
            n_flows=config.model.prior_encoder.flow.n_flows,
            flow_kernel_size=config.model.prior_encoder.flow.kernel_size,
            flow_dilation_rate=config.model.prior_encoder.flow.dilation_rate,
            flow_n_layers=config.model.prior_encoder.flow.n_layers,
            gin_channels=self.speaker_embed_dim,
        )

        self.duration_predictor = StochasticDurationPredictor(
            in_channels=config.model.prior_encoder.hidden_channels,
            filter_channels=config.model.duration_predictor.hidden_channels,
            kernel_size=config.model.duration_predictor.kernel_size,
            p_dropout=config.model.duration_predictor.p_dropout,
            n_flows=config.model.duration_predictor.n_flows,
            gin_channels=self.speaker_embed_dim,
        )

        self.decoder = HiFiGANDecoder(
            initial_channel=self.hidden_channels,
            resblock_kernel_sizes=config.model.decoder.resblock_kernel_sizes,
            resblock_dilation_sizes=config.model.decoder.resblock_dilation_sizes,
            upsample_rates=config.model.decoder.upsample_rates,
            upsample_kernel_sizes=config.model.decoder.upsample_kernel_sizes,
            upsample_initial_channel=config.model.decoder.upsample_initial_channel,
            gin_channels=self.speaker_embed_dim,
        )

    def forward(self, phonemes, phoneme_lengths, mel, mel_lengths, speaker_id=None):
        if self.n_speakers > 1 and speaker_id is not None:
            g = self.emb_g(speaker_id).unsqueeze(-1)
        else:
            g = self.speaker_encoder(mel).unsqueeze(-1)

        z_p, m_p, logs_p, phoneme_mask = self.prior_encoder(phonemes, phoneme_lengths, g=g)

        z_p = torch.nan_to_num(z_p, nan=0.0)
        m_p = torch.nan_to_num(m_p, nan=0.0)
        logs_p = torch.clamp(logs_p, min=-10, max=10)

        z_q, m_q, logs_q, mel_mask = self.posterior_encoder(mel, mel_lengths, g=g)

        z_slice, ids_slice = self.rand_slice_segments(
            z_q, mel_lengths, self.config.data.segment_size // self.config.data.hop_length
        )
        m_q_slice, _ = self.rand_slice_segments_with_ids(
            m_q, ids_slice, self.config.data.segment_size // self.config.data.hop_length
        )
        logs_q_slice, _ = self.rand_slice_segments_with_ids(
            logs_q, ids_slice, self.config.data.segment_size // self.config.data.hop_length
        )

        o = self.decoder(z_slice, g=g)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_q**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_q.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            mel_mask_for_attn = mel_mask[:, :, : z_q.size(2)]
            attn_mask = torch.unsqueeze(phoneme_mask, 2) * torch.unsqueeze(mel_mask_for_attn, -1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)

        logw_ = torch.log(w + 1e-6) * phoneme_mask
        logw = self.duration_predictor(
            z_p.detach(), phoneme_mask, w=logw_, g=g.detach(), reverse=False
        )

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = torch.matmul(attn.squeeze(1), z_p.transpose(1, 2)).transpose(1, 2)

        z_p = torch.nan_to_num(z_p, nan=0.0, posinf=10.0, neginf=-10.0)
        m_p = torch.nan_to_num(m_p, nan=0.0, posinf=10.0, neginf=-10.0)
        logs_p = torch.clamp(logs_p, min=-10, max=10)

        z_p_slice, _ = self.rand_slice_segments_with_ids(
            z_p, ids_slice, self.config.data.segment_size // self.config.data.hop_length
        )
        m_p_slice, _ = self.rand_slice_segments_with_ids(
            m_p, ids_slice, self.config.data.segment_size // self.config.data.hop_length
        )
        logs_p_slice, _ = self.rand_slice_segments_with_ids(
            logs_p, ids_slice, self.config.data.segment_size // self.config.data.hop_length
        )

        return (
            o,
            ids_slice,
            phoneme_mask,
            mel_mask,
            (z_p_slice, z_slice, m_p_slice, logs_p_slice, m_q_slice, logs_q_slice),
            (logw, logw_),
            g,
        )

    def infer(self, phonemes, phoneme_lengths, speaker_id=None, noise_scale=1.0, length_scale=1.0):
        if self.n_speakers > 1 and speaker_id is not None:
            g = self.emb_g(speaker_id).unsqueeze(-1)
        else:
            g = None

        z_p, m_p, logs_p, phoneme_mask = self.prior_encoder(phonemes, phoneme_lengths, g=g)

        w = (
            self.duration_predictor(z_p, phoneme_mask, g=g, reverse=True, noise_scale=noise_scale)
            * length_scale
        )
        w_ceil = torch.ceil(w)

        if w_ceil.dim() == 2:
            duration = w_ceil
            mel_lengths = torch.clamp_min(torch.sum(w_ceil, dim=1), 1).long()
        else:
            duration = w_ceil.squeeze()
            if duration.dim() == 1:
                duration = duration.unsqueeze(0)
            mel_lengths = torch.clamp_min(torch.sum(duration, dim=1), 1).long()

        max_mel_length = int(mel_lengths.max().item())
        mel_mask = torch.unsqueeze(self.sequence_mask(mel_lengths, max_mel_length), 1).to(phoneme_mask.dtype)

        attn_mask = torch.unsqueeze(phoneme_mask, 2) * torch.unsqueeze(mel_mask, -1)
        attn_mask = attn_mask.squeeze(1)

        attn = generate_path(duration, attn_mask).unsqueeze(1)

        z = torch.matmul(attn.squeeze(1).transpose(1, 2), z_p.transpose(1, 2)).transpose(1, 2)
        z = z * mel_mask

        o = self.decoder(z, g=g)
        o = torch.clamp(o, min=-1.0, max=1.0)

        return o, attn, mel_mask, (z, z_p, m_p, logs_p)

    def rand_slice_segments(self, x, x_lengths, segment_size):
        b, d, t = x.size()

        ids_str_max = x_lengths - segment_size + 1
        ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)

        ret = torch.zeros_like(x[:, :, :segment_size])
        for i in range(b):
            idx_str = ids_str[i]
            idx_end = idx_str + segment_size
            ret[i] = x[i, :, idx_str:idx_end]

        return ret, ids_str

    def rand_slice_segments_with_ids(self, x, ids_str, segment_size):
        b, d, t = x.size()
        ret = torch.zeros_like(x[:, :, :segment_size])
        for i in range(b):
            idx_str = ids_str[i]
            idx_end = idx_str + segment_size
            ret[i] = x[i, :, idx_str:idx_end]
        return ret, ids_str

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        if length.dim() == 0:
            length = length.unsqueeze(0)
        batch_size = length.size(0)
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        x = x.unsqueeze(0).expand(batch_size, -1)
        length = length.view(batch_size, 1)
        return x < length
