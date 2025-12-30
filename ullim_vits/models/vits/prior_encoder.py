import torch
import torch.nn as nn
import torch.nn.functional as F
from ullim_vits.models.common.attention import RelativePositionMultiHeadAttention, FeedForward
from ullim_vits.models.common.flows import ResidualCouplingBlock


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder.append(
                TransformerEncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * torch.sqrt(torch.tensor(self.hidden_channels, dtype=torch.float32))
        x = x.transpose(1, 2)

        x_mask = torch.unsqueeze(self.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if g is not None:
            g = self.cond(g)
            x = x + g

        for layer in self.encoder:
            x = layer(x, x_mask)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size, p_dropout):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.attn = RelativePositionMultiHeadAttention(hidden_channels, n_heads, dropout=p_dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.ffn = FeedForward(hidden_channels, filter_channels, dropout=p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        x = x.transpose(1, 2)
        attn_output = self.attn(x, mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        x = x.transpose(1, 2)

        x = x * x_mask

        return x


class PriorEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, n_flows, flow_kernel_size, flow_dilation_rate, flow_n_layers, gin_channels=0):
        super().__init__()

        self.text_encoder = TextEncoder(
            n_vocab, out_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout, gin_channels
        )

        self.flow = ResidualCouplingBlock(
            out_channels, hidden_channels, flow_kernel_size,
            flow_dilation_rate, flow_n_layers, n_flows, gin_channels
        )

    def forward(self, x, x_lengths, g=None):
        x, m, logs, x_mask = self.text_encoder(x, x_lengths, g)

        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        z = self.flow(z, x_mask, g=g, reverse=False)

        return z, m, logs, x_mask
