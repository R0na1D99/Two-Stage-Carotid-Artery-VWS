import torch
from torch import nn
from torch.nn import functional as F


class ResUNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nsf=64, trilinear=False):
        """Res UNet 3D
        cfg:
            in_channels (int): input num of channels
            out_channels (int): output num of channels
            nsf (int, optional): num of base filters. Defaults to 64.
            trilinear (bool, optional): whether use trilinear in decoder up sample, if false use ConvTrans. Defaults to False.
        """        
        super(ResUNet3D, self).__init__()

        self.depth = 4 

        self.encoder1 = ResUNetEncoder(in_channels, nsf)
        self.decoder = ResUNetDecoder(out_channels, nsf, trilinear)

    def forward(self, x: torch.Tensor):
        xs1 = self.encoder1(x)

        out1 = self.decoder(xs1)

        return out1

class ResUNetEncoder(nn.Module):
    def __init__(self, in_channels: int, nsf: int):
        super(ResUNetEncoder, self).__init__()

        self.n_channels = in_channels

        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, nsf, kernel_size=3, padding=1),
            nn.BatchNorm3d(nsf),
            nn.ReLU(inplace=True),
            nn.Conv3d(nsf, nsf, kernel_size=3, padding=1),
            nn.BatchNorm3d(nsf),
        )
        self.in_skip = nn.Sequential(
            nn.Conv3d(in_channels, nsf, kernel_size=3, padding=1),
            nn.BatchNorm3d(nsf),
        )
            
        self.down1 = DoubleResidualConv(1 * nsf, 2 * nsf, stride=(2, 2, 2))
        self.down2 = DoubleResidualConv(2 * nsf, 4 * nsf, stride=(2, 2, 2))
        self.down3 = DoubleResidualConv(4 * nsf, 4 * nsf, stride=(2, 2, 2))

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x) + self.in_skip(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4]


class ResUNetDecoder(nn.Module):
    def __init__(self, out_channels: int, nsf: int, trilinear: bool):
        super(ResUNetDecoder, self).__init__()

        self.up1 = Up(8 * nsf, 2 * nsf, (2, 2, 2), trilinear)
        self.up2 = Up(4 * nsf, 1 * nsf, (2, 2, 2), trilinear)
        self.up3 = Up(2 * nsf, 1 * nsf, (2, 2, 2), trilinear)

        self.outc = nn.Conv3d(nsf, out_channels, kernel_size=1)

    def forward(self, xs):
        x1, x2, x3, x4 = xs
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        return logits


class DoubleResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride=2):
        super(DoubleResidualConv, self).__init__()
        """(BN+ReLU+Conv)*2 + residual(Conv+BN)"""
        self.conv_block = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, scale=2, trilinear=True):
        super().__init__()
        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=scale, stride=scale)

        self.conv = DoubleResidualConv(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out


if __name__ == "__main__":
    net = ResUNet3D(1, 1)
    inputs = torch.rand((1, 1, 32, 32, 32))
    print(net(inputs).shape)
