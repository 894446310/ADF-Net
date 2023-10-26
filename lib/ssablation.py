
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DConv(n_channels, 64)
        self.down1 = DDown(64, 128)
        self.down2 = DDown(128, 256)
        self.down3 = DDown(256, 512)
        self.down4 = DDown(512, 512)
        self.up1 = UUp(1024, 256, bilinear)
        self.up2 = UUp(512, 128, bilinear)
        self.up3 = UUp(256, 64, bilinear)
        self.up4 = UUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        dad1=x
        out = self.outc(x)
        x_v_d0=out
        x_v_d1=x_v_d0
        x_v_d2=x_v_d0
        x_v_d3=x_v_d0
        x_v_d4=x_v_d0
        x_v_d5=x_v_d0
        x_v_d6=x_v_d0
        x_v_d7=x_v_d1
        x_v_d8=x_v_d1
        return out,x_v_d0,x_v_d1,x_v_d2,x_v_d3,x_v_d4,x_v_d5,x_v_d6,x_v_d7,x_v_d8
class DConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    input = torch.randn(1, 3, 224, 224)
    out12 = net(input)
    print(net)