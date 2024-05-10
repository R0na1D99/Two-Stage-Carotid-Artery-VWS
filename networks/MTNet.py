import torch
from torch import nn
from torch.nn import functional as F



class MTNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nsf=64, aligner_branch=2, trilinear=False):
        """Multi-modality Transfer-learning Network.

        :param int in_channels: input num of channels
        :param int out_channels: output num of channels
        :param int nsf: num of base filters, defaults to 64
        :param int aligner_branch:  branches that assigned with aligners in skip-connections, defaults to 2
        :param bool trilinear: whether to use trilinear in decoder upsample layers, defaults to False
        """
        super(MTNet, self).__init__()
        self.align_1_flag = False
        self.align_2_flag = False
        self.forward_branch = 1

        self.encoder_1 = ResUNetEncoder(in_channels, nsf)
        self.encoder_2 = ResUNetEncoder(in_channels, nsf)

        assert aligner_branch in [0, 1, 2, 3], KeyError("aligner branch not supported! available options [0-3]")
        if aligner_branch in [1, 3]:
            self.align_1_flag = True
            self.aligners_1 = self.define_aligners(nsf)

        if aligner_branch in [2, 3]:
            self.align_2_flag = True
            self.aligners_2 = self.define_aligners(nsf)

        self.decoder = ResUNetDecoder(out_channels, nsf, trilinear)

    def define_aligners(self, nsf: int) -> nn.ModuleList:
        """Return aligner class based on give numbers of filters.

        :param int nsf: basic num of filters
        :return nn.ModuleList: aligners in `list`
        """
        aligners = nn.ModuleList(
            [DomainAligner(nsf * 1, 1), DomainAligner(nsf * 2, 1), DomainAligner(nsf * 4, 1), DomainAligner(nsf * 4, 1)]
        )
        return aligners

    def forward(self, x):
        if self.forward_branch == 1:
            xs = self.encoder_1(x)
            if self.align_1_flag:
                xs = [layer(_x) for layer, _x in zip(self.aligners_1, xs)]

        elif self.forward_branch == 2:
            xs = self.encoder_2(x)
            if self.align_2_flag:
                xs = [layer(_x) for layer, _x in zip(self.aligners_2, xs)]
        else:
            raise KeyError(self.forward_branch)

        out = self.decoder(xs)
        return out


class ResUNetEncoder(nn.Module):
    def __init__(self, in_channels, nsf=64):
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

    def forward(self, x):
        x1 = self.in_conv(x) + self.in_skip(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4]


class ResUNetDecoder(nn.Module):
    def __init__(self, out_channels, nsf=64, trilinear=False):
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
        """(bn+ReLU+conv)*2 + residual(conv+bn)"""
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


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: Number of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.ReLU(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: Number of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv3d(num_channels, 1, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            # out = self.conv(input_tensor)
            x = self.pre_conv(input_tensor)
            out = self.conv2(self.conv1(x))

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSEBlock(nn.Module):
    """
    3D extension of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: Number of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSEBlock, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, x):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        x = torch.max(self.cSE(x), self.sSE(x))
        return x
    

class DomainAligner(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: Number of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super().__init__()
        self.conv = DoubleResidualConv(num_channels, num_channels, stride=1)
        self.se = ChannelSpatialSEBlock(num_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        x = self.conv(x)
        return self.se(x)
