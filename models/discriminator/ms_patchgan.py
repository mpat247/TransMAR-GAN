# /discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SingleScaleDiscriminator(nn.Module):
    """
    Single-scale PatchGAN discriminator with spectral normalization.

    Input:  (B, C, H, W)   where C = 2  (e.g., concat[ct_corr, ct_target])
    Output:
        logits:   (B, 1, H_out, W_out)
        features: list of feature maps from intermediate layers for FM loss
    """
    def __init__(
        self,
        in_channels: int = 2,     # ct + real/fake
        base_channels: int = 64,
        num_layers: int = 5,
        use_sn: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        convs = []
        ch_in = in_channels
        ch_out = base_channels

        for i in range(num_layers):
            # First (num_layers-1) layers: stride=2, last layer: stride=1 (PatchGAN-style)
            stride = 1 if i == num_layers - 1 else 2
            conv = nn.Conv2d(
                ch_in, ch_out,
                kernel_size=4,
                stride=stride,
                padding=1
            )
            if use_sn:
                conv = spectral_norm(conv)
            convs.append(conv)

            # Update channels for next layer
            ch_in = ch_out
            if i < num_layers - 2:
                # Double channels up to 8 * base_channels (but not for last two layers)
                ch_out = min(ch_out * 2, base_channels * 8)

        self.convs = nn.ModuleList(convs)
        
        # Final 1x1 conv to reduce to 1 channel for logits
        final_ch = ch_in  # Last channel count from the loop
        self.final_conv = nn.Conv2d(final_ch, 1, kernel_size=1, stride=1, padding=0)
        if use_sn:
            self.final_conv = spectral_norm(self.final_conv)

    def forward(self, x, return_features: bool = True):
        """
        Args:
            x: (B, C, H, W)
            return_features: whether to return intermediate feature maps

        Returns:
            logits:   (B, 1, H_out, W_out)
            features: list[Tensor] of intermediate feature maps (if return_features)
        """
        feats = []
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h)
            # Apply activation to all layers except the very last
            if i != self.num_layers - 1:
                h = self.lrelu(h)
            if return_features and i < self.num_layers - 1:
                feats.append(h)
        
        # Apply final 1x1 conv to get single-channel logits
        logits = self.final_conv(h)  # (B, 1, H', W')
        
        if return_features:
            return logits, feats
        return logits, None


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator: D(1), D(1/2), D(1/4).

    Each scale uses the SAME architecture (SingleScaleDiscriminator),
    but sees progressively downsampled inputs.

    For CT MAR:
        input to D is concat([xcorr, x_real_or_fake], dim=1)  -> 2 channels.
    """
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_layers: int = 5,
        num_scales: int = 3,
        use_sn: bool = True
    ):
        super().__init__()
        self.num_scales = num_scales

        self.discriminators = nn.ModuleList([
            SingleScaleDiscriminator(
                in_channels=in_channels,
                base_channels=base_channels,
                num_layers=num_layers,
                use_sn=use_sn
            )
            for _ in range(num_scales)
        ])

    def forward(self, x, return_features: bool = True):
        """
        Args:
            x: (B, C, H, W)  full-resolution input
            return_features: if True, also returns features per scale

        Returns:
            logits_all:   list of length num_scales,
                         each entry is (B,1,H_s,W_s)
            features_all: list of length num_scales,
                         each entry is list[Tensor] of features for that scale
        """
        logits_all = []
        features_all = [] if return_features else None

        x_scale = x
        for d in self.discriminators:
            logits, feats = d(x_scale, return_features=True)
            logits_all.append(logits)
            if return_features:
                features_all.append(feats)

            # downsample for next scale: 1 -> 1/2 -> 1/4
            x_scale = F.avg_pool2d(
                x_scale,
                kernel_size=2,
                stride=2,
                padding=0
            )

        if return_features:
            return logits_all, features_all
        return logits_all, None

