import torch
import torch.nn as nn
from ullim_vits.models.common.residual import WaveNetResidualBlock


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        self.enc = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** i
            self.enc.append(
                WaveNetResidualBlock(hidden_channels, kernel_size, dilation=dilation, dropout=0.0)
            )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(self.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask

        if g is not None:
            g = self.cond(g)

        for i in range(self.n_layers):
            x = self.enc[i](x, x_mask)
            if g is not None:
                x = x + g
            x = x * x_mask

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)
