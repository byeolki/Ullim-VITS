import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])

        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b
