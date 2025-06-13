import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            t = t + pad_len
        x = x.view(b, c, t // self.period, self.period)
        x = x.permute(0, 1, 3, 2)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(p) for p in [2, 3, 5, 7, 11]
        ])

    def forward(self, x):
        ret = []
        for d in self.discriminators:
            ret.append(d(x))
        return ret

class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(128, 128, 41, 4, groups=4, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(128, 256, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.AvgPool1d(4, 2, padding=2)
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS()
        ])

    def forward(self, x):
        ret = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.pooling(x)
            ret.append(d(x))
        return ret
