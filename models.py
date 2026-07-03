"""Networks for the experiment.

* DeblurNet(use_imu=False) : small 2-level U-Net, blur -> sharp.  (BASELINE)
* DeblurNet(use_imu=True)  : same U-Net + a gyro (IMU) encoder whose embedding
                             modulates the bottleneck via FiLM.  (OURS)

Both are deliberately small — the point is the *delta* from adding IMU, not SOTA.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class GyroEncoder(nn.Module):
    """(B, T, axes) gyro series -> (gamma, beta) FiLM params for `out_ch` channels."""

    def __init__(self, axes: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(axes, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, 2 * out_ch)
        self.out_ch = out_ch

    def forward(self, imu):                 # imu: (B, T, axes)
        h = self.conv(imu.transpose(1, 2))  # (B, 32, 1)
        gb = self.head(h.squeeze(-1))       # (B, 2*out_ch)
        gamma, beta = gb[:, : self.out_ch], gb[:, self.out_ch:]
        return gamma, beta


class DeblurNet(nn.Module):
    def __init__(self, base_channels: int = 32, imu_axes: int = 2, use_imu: bool = False):
        super().__init__()
        c = base_channels
        self.use_imu = use_imu

        self.enc1 = DoubleConv(1, c)
        self.enc2 = DoubleConv(c, 2 * c)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(2 * c, 4 * c)

        if use_imu:
            self.gyro = GyroEncoder(imu_axes, 4 * c)

        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = DoubleConv(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = DoubleConv(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x, imu=None):
        e1 = self.enc1(x)               # (B, c,   H,   W)
        e2 = self.enc2(self.pool(e1))   # (B, 2c,  H/2, W/2)
        b = self.bottleneck(self.pool(e2))  # (B, 4c, H/4, W/4)

        if self.use_imu:
            assert imu is not None, "use_imu=True requires an imu tensor"
            gamma, beta = self.gyro(imu)             # (B, 4c) each
            b = b * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        residual = self.out(d1)
        return torch.clamp(x + residual, 0.0, 1.0)   # predict a residual on the blur


def build_model(cfg: dict, use_imu: bool) -> DeblurNet:
    return DeblurNet(
        base_channels=cfg["train"]["base_channels"],
        imu_axes=cfg["imu"]["axes"],
        use_imu=use_imu,
    )
