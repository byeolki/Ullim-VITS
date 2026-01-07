import torch
import torch.nn as nn
import torch.nn.functional as F


class ActNorm(nn.Module):
    def __init__(self, channels, ddi=False):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, g=None, reverse=False):
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = self.logs + self.bias + x * torch.exp(self.logs)
            if x_mask is not None:
                z = z * x_mask

            logdet = torch.sum(self.logs) * torch.sum(x_mask, [1, 2])

        return z, logdet

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            self.logs.data.copy_(logs.view(1, -1, 1))
            self.bias.data.copy_((-m * torch.exp(-logs)).view(1, -1, 1))


class CouplingBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not reverse:
            x1 = x1 + stats
            logdet = 0
        else:
            x1 = x1 - stats
            logdet = None

        x = torch.cat([x0, x1], 1)
        return x, logdet


class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, dropout=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)

            in_layer = nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                dilation=dilation, padding=padding)
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)
            g = torch.clamp(g, min=-10, max=10)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = torch.clamp(x_in, min=-100, max=100)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = x_in + g_l
            acts = acts * x_mask
            acts = torch.clamp(acts, min=-100, max=100)

            acts = F.glu(acts, dim=1)

            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(CouplingBlock(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels))
            self.flows.append(ActNorm(channels, ddi=True))

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)

        return x
