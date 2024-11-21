import torch
import torch.nn as nn

# Implementation based on https://github.com/milesial/Pytorch-UNet/tree/master


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_filters: int = 32,
        depth: int = 4,
    ):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.depth = depth

        # Input layer
        self.inn = UNetDoubleConv(in_channels, n_filters)

        # Encoder path (down-sampling)
        self.down_layers = nn.ModuleList()
        for level in range(depth):
            self.down_layers.append(
                UNetDown(n_filters * 2**level, n_filters * 2 ** (level + 1))
            )

        # Decoder path (up-sampling)
        self.up_layers = nn.ModuleList()
        for level in range(depth, 0, -1):
            self.up_layers.append(
                UNetUp(n_filters * 2**level, n_filters * 2 ** (level - 1))
            )

        # Output layer
        self.out = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x1 = self.inn(x)

        # Encoder path
        encoder_outputs = [x1]
        for down in self.down_layers:
            x = down(encoder_outputs[-1])
            encoder_outputs.append(x)

        # Decoder path with skip connections
        for level, up in enumerate(self.up_layers):
            x = up(x, encoder_outputs[self.depth - 1 - level])

        return self.out(x)


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetDoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


class UNetDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(UNetDown, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetDoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.maxpool_conv(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetUp, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Dropout2d(), UNetDoubleConv(in_channels, out_channels)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)

        # Compute necessary padding for x1 to match x2 shape
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2 shape
        x1 = nn.functional.pad(
            x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )

        # Concatenate x2 and x1 along the channels dimension
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
