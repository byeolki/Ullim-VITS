import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(x, 0.1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x)
        return x + residual


class WaveNetResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        self.conv = nn.Conv1d(channels, 2*channels, kernel_size,
                             padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, x, g=None):
        residual = x

        x = self.conv(x)

        if g is not None:
            x = x + g

        gate, filter_out = torch.chunk(x, 2, dim=1)
        x = torch.sigmoid(gate) * torch.tanh(filter_out)

        x = self.projection(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        return (x + residual) * 0.7071


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=dilation[0], padding=dilation[0]*(kernel_size-1)//2),
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=dilation[1], padding=dilation[1]*(kernel_size-1)//2),
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=dilation[2], padding=dilation[2]*(kernel_size-1)//2)
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=1, padding=(kernel_size-1)//2),
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=1, padding=(kernel_size-1)//2),
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=1, padding=(kernel_size-1)//2)
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=dilation[0], padding=dilation[0]*(kernel_size-1)//2),
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=dilation[1], padding=dilation[1]*(kernel_size-1)//2)
        ])

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x
