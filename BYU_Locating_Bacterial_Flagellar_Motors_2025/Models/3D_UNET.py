import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet3D, self).__init__()
        self.encoder1 = DoubleConv3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv3D(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv3D(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv3D(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = DoubleConv3D(256, 512)

        self.up4 = UpConv3D(512, 256)
        self.decoder4 = DoubleConv3D(512, 256)  # skip concat 후 (256+256)

        self.up3 = UpConv3D(256, 128)
        self.decoder3 = DoubleConv3D(256, 128)  # skip concat 후 (128+128)

        self.up2 = UpConv3D(128, 64)
        self.decoder2 = DoubleConv3D(128, 64)   # skip concat 후 (64+64)

        self.up1 = UpConv3D(64, 32)
        self.decoder1 = DoubleConv3D(64, 32)    # skip concat 후 (32+32)

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)     # (B, 32, D, H, W)
        p1 = self.pool1(e1)       # (B, 32, D/2, H/2, W/2)

        e2 = self.encoder2(p1)    # (B, 64, ...)
        p2 = self.pool2(e2)       # (B, 64, D/4, H/4, W/4)

        e3 = self.encoder3(p2)    # (B, 128, ...)
        p3 = self.pool3(e3)       # (B, 128, D/8, H/8, W/8)

        e4 = self.encoder4(p3)    # (B, 256, ...)
        p4 = self.pool4(e4)       # (B, 256, D/16, H/16, W/16)

        # bottleneck
        b = self.bottleneck(p4)   # (B, 512, D/16, H/16, W/16)

        up4 = self.up4(b)         # (B, 256, D/8, H/8, W/8)
        cat4 = torch.cat([up4, e4], dim=1)  # (B, 256+256, ...)
        d4 = self.decoder4(cat4)  # (B, 256, ...)

        up3 = self.up3(d4)        # (B, 128, D/4, H/4, W/4)
        cat3 = torch.cat([up3, e3], dim=1)  # (B, 128+128, ...)
        d3 = self.decoder3(cat3)  # (B, 128, ...)

        up2 = self.up2(d3)        # (B, 64, D/2, H/2, W/2)
        cat2 = torch.cat([up2, e2], dim=1)  # (B, 64+64, ...)
        d2 = self.decoder2(cat2)  # (B, 64, ...)

        up1 = self.up1(d2)        # (B, 32, D, H, W)
        cat1 = torch.cat([up1, e1], dim=1)  # (B, 32+32, D, H, W)
        d1 = self.decoder1(cat1)  # (B, 32, D, H, W)

        out = self.out_conv(d1)   # (B, out_channels, D, H, W)
        return out