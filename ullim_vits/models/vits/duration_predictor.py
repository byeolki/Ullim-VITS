import torch
import torch.nn as nn
import torch.nn.functional as F
from ullim_vits.models.common.flows import ResidualCouplingBlock
from ullim_vits.utils.alignment import generate_path


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(2):
            self.convs.append(
                nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
            )
            self.norms.append(nn.LayerNorm(filter_channels))

        self.post_pre = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)

        self.post_convs = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        for _ in range(2):
            self.post_convs.append(
                nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
            )
            self.post_norms.append(nn.LayerNorm(filter_channels))

        self.post_flows = ResidualCouplingBlock(
            1, filter_channels, kernel_size, 1, 4, n_flows, gin_channels
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = self.pre(x)

        if g is not None:
            g_exp = self.cond(g)
            x = x + g_exp

        x = x * x_mask

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = F.relu(x)
            x = F.dropout(x, self.p_dropout, self.training)

        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.post_pre(x)

            for conv, norm in zip(self.post_convs, self.post_norms):
                flows = conv(flows)
                flows = norm(flows.transpose(1, 2)).transpose(1, 2)
                flows = F.relu(flows)
                flows = F.dropout(flows, self.p_dropout, self.training)

            flows = self.post_proj(flows) * x_mask

            logw = w.unsqueeze(1)
            logw = self.post_flows(logw, x_mask, g=flows, reverse=reverse)
            logw = logw.squeeze(1)

            return logw
        else:
            flows = self.post_pre(x)

            for conv, norm in zip(self.post_convs, self.post_norms):
                flows = conv(flows)
                flows = norm(flows.transpose(1, 2)).transpose(1, 2)
                flows = F.relu(flows)
                flows = F.dropout(flows, self.p_dropout, self.training)

            flows = self.post_proj(flows) * x_mask

            logw = torch.randn(x.size(0), 1, x.size(2)).to(x.dtype).to(x.device) * noise_scale
            logw = self.post_flows(logw, x_mask, g=flows, reverse=reverse)
            logw = logw.squeeze(1)

            w = torch.exp(logw) * x_mask.squeeze(1)

            return w


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(2):
            self.convs.append(
                nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
            )
            self.norms.append(nn.LayerNorm(filter_channels))

        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)

        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x * x_mask)
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = F.relu(x)
            x = F.dropout(x, self.p_dropout, self.training)

        x = self.proj(x * x_mask)

        return x * x_mask
