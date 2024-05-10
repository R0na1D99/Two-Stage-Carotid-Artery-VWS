import os

import torch
import argparse
import torch.nn as nn
from monai.networks import nets
from .ResUNet3D import ResUNet3D
from torch.optim import SGD, Adam, AdamW, lr_scheduler, Optimizer


def define_optimizer(opt_name: str, params, lr, **args):
    if opt_name == "adam":
        return Adam(params, lr, **args)
    elif opt_name == "adamw":
        return AdamW(params, lr, **args)
    elif opt_name == "sgd":
        return SGD(params, lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise NotImplementedError(opt_name)


def define_scheduler(optimizer: Optimizer, scheduler_type: str, max_epochs: int):
    if scheduler_type == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif scheduler_type == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)
    elif scheduler_type == "lambda":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)
    else:
        return NotImplementedError(scheduler_type)
    return scheduler


def define_S(spatial_dims: int, cfg: argparse.Namespace):
    nsf = 32
    if spatial_dims == 3 and cfg.netS == "resunet":  # better comparison with same implementation
        net = ResUNet3D(cfg.in_channels, cfg.out_channels, cfg.nsf, trilinear=False)
    else:
        channels = [nsf, nsf * 2, nsf * 4, nsf * 8, nsf * 16]
        layers = 4 if spatial_dims == 3 else 5  # GPU memory limitation
        if cfg.netS == "unet":
            net = nets.BasicUNet(
                spatial_dims,
                cfg.in_channels,
                cfg.out_channels,
                features=channels[:layers] + [nsf],
                act="relu",
                norm="batch",
            )
        elif cfg.netS == "resunet":
            net = nets.UNet(
                spatial_dims=spatial_dims,
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                channels=channels[:layers],
                strides=[2] * layers,
                num_res_units=2,
                norm="batch",
            )
        elif cfg.netS == "unetr":
            net = nets.UNETR(
                spatial_dims=spatial_dims,
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                img_size=cfg.roi_size ,
                feature_size=24,
            )
        elif cfg.netS == "swinunetr":
            net = nets.SwinUNETR(
                cfg.roi_size ,
                cfg.in_channels,
                cfg.out_channels,
                feature_size=24,
                spatial_dims=spatial_dims,
            )
        elif cfg.netS == "attentionunet":
            net = nets.AttentionUnet(
                spatial_dims,
                cfg.in_channels,
                cfg.out_channels,
                channels=channels[:layers],
                strides=[2] * layers,
            )
        elif cfg.netS == "unetplusplus":
            net = nets.BasicUNetPlusPlus(
                spatial_dims,
                cfg.in_channels,
                cfg.out_channels,
                channels[:layers] + [nsf],
                deep_supervision=True,
            )
    return init_weights(net, cfg.init_type, cfg.init_gain)


def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm3d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net
