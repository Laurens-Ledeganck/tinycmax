from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LazyConvGru(nn.Module):
    """
    Lazy convolutional GRU layer following PyTorch convention,
    and concatenating gates instead of input/hidden.
    """

    def __init__(self, out_channels, kernel_size, padding_mode="zeros"):
        super().__init__()

        self.out_channels = out_channels
        padding = kernel_size // 2

        self.ih = nn.LazyConv2d(3 * out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.hh = nn.LazyConv2d(3 * out_channels, kernel_size, padding=padding, padding_mode=padding_mode)

    def forward(self, x, hx):
        # get previous output
        if hx is None:
            b, _, h, w = x.shape
            hx = torch.zeros(b, self.out_channels, h, w, device=x.device, dtype=x.dtype)

        # compute concatenated gates
        ih, hh = self.ih(x), self.hh(hx)
        ih_r, ih_z, ih_n = ih.chunk(3, dim=1)
        hh_r, hh_z, hh_n = hh.chunk(3, dim=1)
        r = torch.sigmoid(ih_r + hh_r)
        z = torch.sigmoid(ih_z + hh_z)
        n = torch.tanh(ih_n + r * hh_n)

        # compute hidden
        hx = (1 - z) * hx + z * n
        return hx


class LazyConvMinGru(nn.Module):
    """
    From 'Were RNNs All We Needed?' (Feng et al., arXiv'24).
    Code adapted from https://github.com/lucidrains/minGRU-pytorch/blob/main/minGRU_pytorch/minGRU.py.
    """

    def __init__(self, out_channels, kernel_size, padding_mode="zeros"):
        super().__init__()

        self.out_channels = out_channels
        padding = kernel_size // 2

        self.ih = nn.LazyConv2d(2 * out_channels, kernel_size, padding=padding, padding_mode=padding_mode)

    def forward(self, x, hx):
        # compute hidden and gate
        h_, z = self.ih(x).chunk(2, dim=1)
        z = z.sigmoid()

        # update hidden
        hx = torch.lerp(hx, h_, z) if hx is not None else h_  # (1 - z) * hx + z * h_
        return hx


class Residual(nn.Sequential):
    """
    Residual block with connection from input to after.
    """

    def forward(self, input):
        x = input
        *before, residual, after = self
        for layer in before:
            x = layer(x)
        res = residual(input)
        x = after(x + res)
        return x


def feedforward(synapse, neuron):
    return nn.Sequential(
        OrderedDict(
            [
                ("synapse", synapse),
                ("neuron", neuron),
            ]
        )
    )


def named_sequential(prefix, *args):
    modules = OrderedDict([(f"{prefix}{i}", arg) for i, arg in enumerate(args)])
    return nn.Sequential(modules)


def named_residual(prefix, *args):
    modules = OrderedDict([(f"{prefix}{i}", arg) for i, arg in enumerate(args)])
    return Residual(modules)


def res_block(out_channels, kernel_size, activation_fn, stride=1, padding_mode="zeros"):
    padding = kernel_size // 2
    if stride != 1:
        downsample = feedforward(
            nn.LazyConv2d(out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
            nn.Identity(),
        )
    else:
        downsample = nn.Identity()
    block = named_residual(
        "res",
        feedforward(
            nn.LazyConv2d(out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
            activation_fn(),
        ),
        feedforward(
            nn.LazyConv2d(out_channels, kernel_size, padding=padding, padding_mode=padding_mode),
            nn.Identity(),
        ),
        downsample,
        activation_fn(),
    )
    return block


class LazyPadder(nn.Module):
    """
    Pad to size divisible by factor.
    """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        _, _, h, w = x.shape
        pad_h = (self.factor - h % self.factor) % self.factor
        pad_w = (self.factor - w % self.factor) % self.factor
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


def conv_encoder(out_channels, activation_fn, padding_mode="zeros"):
    """
    Components:
    - Padding to size divisible by 8
    - Head with large kernel
    - 2 pairs of residual blocks with stride
    """
    padder = LazyPadder(8)
    head = feedforward(
        nn.LazyConv2d(out_channels // 4, 7, stride=2, padding=3, padding_mode=padding_mode),
        activation_fn(),
    )
    encoder = named_sequential(
        "conv",
        res_block(out_channels // 2, 3, activation_fn, stride=2, padding_mode=padding_mode),
        res_block(out_channels, 3, activation_fn, stride=2, padding_mode=padding_mode),
    )
    return named_sequential("enc", padder, head, encoder)


def upsample_decoder(out_channels, activation_fn, final_bias, padding_mode="zeros", mode="flow"):
    """
    Select between flow (2-channel, identity)
    and depth/disparity (1-channel, softplus/sigmoid) decoder.
    """
    final_channels = 2 if mode == "flow" else 1
    if mode == "flow":
        final_activation = nn.Identity()
    elif mode == "depth":
        final_activation = nn.Softplus()
    elif mode == "disparity":
        final_activation = nn.Sigmoid()
    decoder = named_sequential(
        "dec",
        feedforward(
            nn.LazyConv2d(out_channels, 3, padding=1, padding_mode=padding_mode),
            activation_fn(),
        ),
        feedforward(
            nn.LazyConv2d(final_channels, 3, padding=1, bias=final_bias, padding_mode=padding_mode),
            final_activation,
        ),
        nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
    )
    return decoder
