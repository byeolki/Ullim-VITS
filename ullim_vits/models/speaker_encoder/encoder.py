import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=256, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        in_channels = input_dim
        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else hidden_dim
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)

        self.gru.flatten_parameters()
        _, h = self.gru(x)

        h = h.squeeze(0)

        embedding = self.projection(h)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class ReferenceEncoder(nn.Module):
    def __init__(self, mel_channels=80, hidden_dim=256, output_dim=256):
        super().__init__()
        self.mel_channels = mel_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(128),
        ])

        self.gru = nn.GRU(128 * (mel_channels // 64), hidden_dim, batch_first=True)

        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, time, -1)

        self.gru.flatten_parameters()
        _, h = self.gru(x)

        h = h.squeeze(0)

        embedding = self.projection(h)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
