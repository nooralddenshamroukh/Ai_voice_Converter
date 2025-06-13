import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation * (kernel_size // 2), dilation=dilation))
            for dilation in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation * (kernel_size // 2), dilation=dilation))
            for dilation in dilations
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class HiFiGANGenerator(nn.Module):
    def __init__(self, input_channels=769):
        super().__init__()
        self.num_kernels = 3
        self.num_upsamples = 4

        self.conv_pre = nn.utils.weight_norm(nn.Conv1d(input_channels, 512, 7, 1, padding=3))

        self.upsample_rates = [8, 8, 2, 2]
        self.upsample_kernel_sizes = [16, 16, 4, 4]
        self.upsample_initial_channel = 512

        self.resblock_kernel_sizes = [3, 5, 7]
        self.resblock_dilation_sizes = [
            [1, 3, 5],
            [1, 3, 5],
            [1, 3, 5],
        ]

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        self.upsample_initial_channel // (2 ** i),
                        self.upsample_initial_channel // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2
                    )
                )
            )
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for _ in range(self.num_kernels):
                self.resblocks.append(ResBlock1(ch, self.resblock_kernel_sizes[_], self.resblock_dilation_sizes[_]))

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = []
            for j in range(self.num_kernels):
                xs.append(self.resblocks[i * self.num_kernels + j](x))
            x = sum(xs) / len(xs)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
