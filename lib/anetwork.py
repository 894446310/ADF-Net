"""
这是根据UNet模型搭建出的一个基本网络结构
输入和输出大小是一样的，可以根据需求进行修改
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
# class UNet(nn.Module):
#
#     def __init__(self):
#         super(UNet, self).__init__()
#
#         # 4次下采样
#         self.C1 = Conv(3, 64)
#         self.D1 = DownSampling(64)
#         self.C2 = Conv(64, 128)
#         self.D2 = DownSampling(128)
#         self.C3 = Conv(128, 256)
#         self.D3 = DownSampling(256)
#         self.C4 = Conv(256, 512)
#         self.D4 = DownSampling(512)
#         self.C5 = Conv(512, 1024)
#
#         # 4次上采样
#         self.U1 = UpSampling(1024)
#         self.C6 = Conv(1024, 512)
#         self.U2 = UpSampling(512)
#         self.C7 = Conv(512, 256)
#         self.U3 = UpSampling(256)
#         self.C8 = Conv(256, 128)
#         self.U4 = UpSampling(128)
#         self.C9 = Conv(128, 64)
#
#         self.Th = torch.nn.Sigmoid()
#         self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)
#
#     def forward(self, x):
#         # 下采样部分
#         R1 = self.C1(x)
#         R2 = self.C2(self.D1(R1))
#         R3 = self.C3(self.D2(R2))
#         R4 = self.C4(self.D3(R3))
#         Y1 = self.C5(self.D4(R4))
#
#         # 上采样部分
#         # 上采样的时候需要拼接起来
#         O1 = self.C6(self.U1(Y1, R4))
#         O2 = self.C7(self.U2(O1, R3))
#         O3 = self.C8(self.U3(O2, R2))
#         O4 = self.C9(self.U4(O3, R1))
#
#         # 输出预测，这里大小跟输入是一致的
#         # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
#         return self.pred(O4)
#
#
# if __name__ == '__main__':
#     a = torch.randn(2, 3, 256, 256)
#     net = UNet()
#     print(net(a).shape)
class conv_block(nn.Module):                                        #卷积块
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),

            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):                                           #上采样
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),

			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):  # 注意力模块  #shangcai,yuanlai
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)  # 第一次卷积通道数为64

        # self.root = root_block(ch_in=64, ch_out=64)
        # self.Conv6 = conv_block(ch_in=64, ch_out=64)

        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # 第一次两层卷积通道数64（1➡64）

        '''x = self.Conv1(x)
        x1 = self.Conv6(x)
        x1 += x'''

        x2 = self.Maxpool(x1)  # 第一次下采样
        x2 = self.Conv2(x2)  # 第二次

        x3 = self.Maxpool(x2)  # 第二次下采样
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)  # 第三次下采样
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)  # 第四次下采样
        x5 = self.Conv5(x5)  # 输入512通道数，输出1024（512➡1024）

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


from functools import partial
# from .resnet import resnet34
from torchvision.models import resnet50

# import Constants
from torchvision import models
nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out
class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class CE_Net(nn.Module):
    def __init__(self, classes=1, channels=3):
        super(CE_Net, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        e1 = self.encoder1(x)     # [2, 64, 56, 80]
        e2 = self.encoder2(e1)    # [2, 128, 28, 40]
        e3 = self.encoder3(e2)    # [2, 256, 14, 20]
        e4 = self.encoder4(e3)    # [2, 512, 7, 10]

        # # Center
        e4 = self.dblock(e4)      # [2, 512, 7, 10]
        e4 = self.spp(e4)
        dd4 = self.decoder4(e4)# [2, 516, 7, 10]

        # Decoder
        d4 = self.decoder4(e4) + e3  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d3 = self.decoder3(d4) + e2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d2 = self.decoder2(d3) + e1  # [2, 64, 56, 80]
        d1 = self.decoder1(d2)       # [2, 64, 112, 160]

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # [2, 2, 224, 320]

        return out


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight.data)
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""

import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
# import torchsummary
from torch.nn import init

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight.data)
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CPF_Net(nn.Module):
    def __init__(self, classes=1, channels=3, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                 base_channel=32):
        super(CPF_Net, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        # self.backbone =resnet34(pretrained=False)
        self.expansion = expansion
        self.base_channel = base_channel
        if self.expansion == 4 and self.base_channel == 64:
            expan = [512, 1024, 2048]
            spatial_ch = [128, 256]
        elif self.expansion == 4 and self.base_channel == 32:
            expan = [256, 512, 1024]
            spatial_ch = [32, 128]
            conv_channel_up = [256, 384, 512]
        elif self.expansion == 2 and self.base_channel == 32:
            expan = [128, 256, 512]
            spatial_ch = [64, 64]
            conv_channel_up = [128, 256, 512]

        conv_channel = expan[0]

        self.is_training = is_training
        self.sap = SAPblock(expan[-1])

        self.decoder5 = DecoderBlock(expan[-1], expan[-2], relu=False, last=True)  # 256
        self.decoder4 = DecoderBlock(expan[-2], expan[-3], relu=False)  # 128
        self.decoder3 = DecoderBlock(expan[-3], spatial_ch[-1], relu=False)  # 64
        self.decoder2 = DecoderBlock(spatial_ch[-1], spatial_ch[-2])  # 32

        self.mce_2 = GPG_2([spatial_ch[-1], expan[0], expan[1], expan[2]], width=spatial_ch[-1], up_kwargs=up_kwargs)
        self.mce_3 = GPG_3([expan[0], expan[1], expan[2]], width=expan[0], up_kwargs=up_kwargs)
        self.mce_4 = GPG_4([expan[1], expan[2]], width=expan[1], up_kwargs=up_kwargs)

        self.main_head = BaseNetHead(spatial_ch[0], classes, 2,
                                     is_aux=False, norm_layer=norm_layer)

        self.relu = nn.ReLU()

    #    initialize_weights(self)

    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64

        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512
        m2 = self.mce_2(c2, c3, c4, c5)
        m3 = self.mce_3(c3, c4, c5)
        m4 = self.mce_4(c4, c5)
        # d_bottom=self.bottom(c5)
        c5 = self.sap(c5)

        # d5=d_bottom+c5           #512

        d4 = self.relu(self.decoder5(c5) + m4)  # 256
        d3 = self.relu(self.decoder4(d4) + m3)  # 128
        d2 = self.relu(self.decoder3(d3) + m2)  # 64
        d1 = self.decoder2(d2) + c1  # 32
        main_out = self.main_head(d1)

        final = main_out

        return final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)


class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
        feat = self.conv_out(feat)
        return feat


class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(2 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(2 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat)], dim=1)
        feat = self.conv_out(feat)
        return feat


class GPG_2(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)
        feat = self.conv_out(feat)
        return feat


class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear',
                              align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output


class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3,
                                 padding=1)

        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.conv3x3_1 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)])
        self.conv3x3_2 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)])
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0, dilation=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_size = x.size()

        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)

        branches_2 = F.conv2d(x, self.conv3x3.weight, padding=2, dilation=2)  # share weight
        branches_2 = self.bn[1](branches_2)

        branches_3 = F.conv2d(x, self.conv3x3.weight, padding=4, dilation=4)  # share weight
        branches_3 = self.bn[2](branches_3)

        feat = torch.cat([branches_1, branches_2], dim=1)
        # feat=feat_cat.detach()
        feat = self.relu(self.conv1x1[0](feat))
        feat = self.relu(self.conv3x3_1[0](feat))
        att = self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)

        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2

        feat1 = torch.cat([fusion_1_2, branches_3], dim=1)
        # feat=feat_cat.detach()
        feat1 = self.relu(self.conv1x1[0](feat1))
        feat1 = self.relu(self.conv3x3_1[0](feat1))
        att1 = self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)

        att_1_2 = att1[:, 0, :, :].unsqueeze(1)
        att_3 = att1[:, 1, :, :].unsqueeze(1)

        ax = self.relu(self.gamma * (att_1_2 * fusion_1_2 + att_3 * branches_3) + (1 - self.gamma) * x)
        ax = self.conv_last(ax)

        return ax


class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.sap = SAPblock(in_planes)
        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last == False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs



import torch
from torchvision import models as resnet_model
from torch import nn


class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

# from .DeiT import deit_small_patch16_224 as deit
class FAT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1,pretrained=False):
        super(FAT_Net, self).__init__()

        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        # transformer = deit(pretrained=pretrained)
        # self.transformer=transformer
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.patch_embed = transformer.patch_embed
        # self.transformers = nn.ModuleList(
        #     [transformer.blocks[i] for i in range(12)]
        # )

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf=self.transformer(x)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out


import torch.nn as nn
import torch.nn.functional as F
import torch




class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


class PFC(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, padding=kernel_size // 2),
            # nn.Conv2d(3, channels, kernel_size=3, padding= 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x


# inherit nn.module
class DCSAU(nn.Module):
    def __init__(self, img_channels=3, n_classes=1):
        super(DCSAU, self).__init__()
        self.pfc = PFC(64)
        self.img_channels = img_channels
        self.n_classes = n_classes
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
        self.up_conv1 = Up()
        self.up_conv2 = Up()
        self.up_conv3 = Up()
        self.up_conv4 = Up()
        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4
        self.up1 = csa_block.layer5
        self.up2 = csa_block.layer6
        self.up3 = csa_block.layer7
        self.up4 = csa_block.layer8

    def forward(self, x):
        x1 = self.pfc(x)
        x2 = self.maxpool(x1)

        x3 = self.down1(x2)
        x4 = self.maxpool(x3)

        x5 = self.down2(x4)
        x6 = self.maxpool(x5)

        x7 = self.down3(x6)
        x8 = self.maxpool(x7)

        x9 = self.down4(x8)

        x10 = self.up_conv1(x9, x7)

        x11 = self.up1(x10)

        x12 = self.up_conv2(x11, x5)
        x13 = self.up2(x12)

        x14 = self.up_conv3(x13, x3)
        x15 = self.up3(x14)

        x16 = self.up_conv4(x15, x1)
        x17 = self.up4(x16)

        x18 = self.out_conv(x17)

        # x19 = torch.sigmoid(x18)
        return x18


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=3,
                 effective_step=[2]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs[0]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_my(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels):
        super().__init__()

        self.up_my = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up_my(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class BCDU_net_D3(nn.Module):
    def __init__(self, classes=1, channels=3):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(BCDU_net_D3, self).__init__()
        self.out_size = (224, 320)

        self.inc = DoubleConv(channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256, dropout=0.3)
        self.down3 = Down(256, 512, dropout=0.3)
        self.flat1 = DoubleConv(512, 512, dropout=0.3)
        self.flat2 = DoubleConv(512 * 2, 512, dropout=0.3)

        self.up3 = Up_my(512)
        self.convlstm3 = ConvLSTM(input_channels=256 * 2, hidden_channels=[128, 128, 256]).cuda()
        self.dconv3 = DoubleConv(256, 256)

        self.up2 = Up_my(256)
        self.convlstm2 = ConvLSTM(input_channels=128 * 2, hidden_channels=[64, 64, 128]).cuda()
        self.dconv2 = DoubleConv(128, 128)

        self.up1 = Up_my(128)
        self.convlstm1 = ConvLSTM(input_channels=64 * 2, hidden_channels=[32, 32, 64]).cuda()
        self.dconv1 = DoubleConv(64, 64)

        self.outc = OutConv(64, classes)
        initialize_weights(self)

    def forward(self, x):
        x1 = self.inc(x)  # [12, 3, 224, 320]  --> [12, 64, 224, 320]
        x2 = self.down1(x1)  # [12, 128, 112, 160]
        x3 = self.down2(x2)  # [12, 256, 56, 80]
        x4 = self.down3(x3)  # [12, 512, 28, 40]

        x5 = self.flat1(x4)  # [12, 512, 28, 40]
        x5_dense = torch.cat([x4, x5], dim=1)  # [12, 1024, 28, 40]
        x6 = self.flat2(x5_dense)  # [12, 512, 28, 40]

        x7 = self.up3(x6, x3)  # [12, 512, 56, 80]
        x8 = self.convlstm3(x7)  #
        x9 = self.dconv3(x8)

        x10 = self.up2(x9, x2)
        x11 = self.convlstm2(x10)
        x12 = self.dconv2(x11)

        x13 = self.up1(x12, x1)
        x14 = self.convlstm1(x13)
        x15 = self.dconv1(x14)

        out = self.outc(x15)


        return out


# if __name__ == '__main__':
#     input = torch.rand(2, 3, 224, 320)
#     model = BCDU_net_D3()
#     out12 = model(input)
#     print(out12.shape)


import time
def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlockee(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_rate=6, se=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_rate = expansion_rate
        self.se = se

        expansion_channels = in_channels * expansion_rate
        se_channels = max(1, int(in_channels * 0.25))

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            print("Error: unsupported kernel size")

        # Expansion
        if expansion_rate != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU()
            )

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=expansion_channels, bias=False),
            nn.BatchNorm2d(expansion_channels),
            nn.ReLU()
        )

        # Squeeze and excitation block
        if se:
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=expansion_channels, out_channels=se_channels, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=se_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):

        x = inputs

        if self.expansion_rate != 1:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)

        if self.se:
            x = self.se_block(x) * x

        x = self.pointwise_conv(x)

        if self.in_channels == self.out_channels and self.stride == 1:
            x = x + inputs

        return x


class EffUNet(nn.Module):
    """ U-Net with EfficientNet-B0 encoder """

    def __init__(self, in_channels=3, classes=1):
        super().__init__()

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.down_block_2 = nn.Sequential(
            MBConvBlock(32, 16, kernel_size=3, stride=1, expansion_rate=1),
            MBConvBlock(16, 24, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.down_block_3 = nn.Sequential(
            MBConvBlock(24, 40, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock(40, 40, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_4 = nn.Sequential(
            MBConvBlock(40, 80, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 112, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_5 = nn.Sequential(
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 192, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 320, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.up_block_4 = DecoderBlockee(432, 256)

        self.up_block_3 = DecoderBlockee(296, 128)

        self.up_block_2 = DecoderBlockee(152, 64)

        self.up_block_1a = DecoderBlockee(96, 32)

        self.up_block_1b = DecoderBlockee(32, 16)

        self.head_conv = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.start_conv(x)

        x2 = self.down_block_2(x1)

        x3 = self.down_block_3(x2)

        x4 = self.down_block_4(x3)

        x5 = self.down_block_5(x4)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x4], dim=1)

        x5 = self.up_block_4(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x3], dim=1)

        x5 = self.up_block_3(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x2], dim=1)

        x5 = self.up_block_2(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x1], dim=1)

        x5 = self.up_block_1a(x5)
        x5 = F.interpolate(x5, scale_factor=2)

        x5 = self.up_block_1b(x5)
        output = self.head_conv(x5)

        return output





from ptflops import get_model_complexity_info
if __name__ == '__main__':
    # input = torch.rand(2, 3, 256, 256)
    # model = BCDU_net_D3()
    # out12 = model(input)
    # print(out12.shape)

    net = DCSAU() #可以为自己搭建的模型
    flops, params = get_model_complexity_info(net, (3,192,256), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)






