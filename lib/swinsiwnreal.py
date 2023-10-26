# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from torch.nn.parameter import Parameter
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models import resnet34 as resnet
#from mmcv_custom import load_checkpoint
#from mmseg.utils import get_root_logger
#from ..builder import BACKBONES
from models.layers.modules import UnetDsv3
from models.layers.scale_attention_layer import scale_atten_convblock
# import apex

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel=64, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # xn = self.avg_pool(x)
        # xn = self.cweight * xn + self.cbias
        # xn = x * self.sigmoid(xn)
        # w_x=xn

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # gs = self.gn(g)
        # gs = self.sweight * gs + self.sbias
        # w_g = g * self.sigmoid(gs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(4, channel, bias=False),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y01=self.avg_pool(x)
        y02=y01.view(b, c)
        y = self.avg_pool(x).view(b, c)
        y1=self.fc(y)
        y12=y1.view(b, c, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)
#coor gaigai
class h_swish(nn.Module):
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x


class am(nn.Module):
    def __init__(self,ch_in):
        super(am, self).__init__()

        self.F1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(ch_in),
            nn.ReLU()
        )


        # self.W4 = nn.Linear(1, 1)
        self.W4 = Conv(1, 1, 3, bn=True, relu=True)
        self.W5 = Conv(1, 1, 3, bn=True, relu=True)

        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

    def forward(self, x):
        U1 = self.F1(x)
        U2 = self.F2(x)
        U = U1 + U2
        p = U

        # spatial attention for cnn branch
        p_in = p
        p = self.compress(p)  # 256通道压缩成2通道，通道压缩
        g1 = p
        p = self.spatial(p)  # 2再成1
        g2 = p


        z1 = self.W4(p)
        # z2 = nn.ReLU()(p)
        z2 = self.W5(p)
        a1 = (torch.exp(z1) / (torch.exp(z1) + torch.exp(z2)))
        a2 = (torch.exp(z2) / (torch.exp(z1) + torch.exp(z2)))

        V = U1 * a1 + U2 * a2 + x
        return V


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)


    def forward(self, x):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_atten(y)
        y = self.upsample(y)
        return (y * x_res) + y


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
#
# class BiFusion_block(nn.Module):
#     def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
#                  drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
#         super(BiFusion_block, self).__init__()
#
#         # channel attention for F_g, use SE Block  ch1是cnn，ch2是transformer   ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256,
#         self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)  # //除完取整
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#         #
#         self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv = conv_block(ch_2, ch_2)
#         self.conv_atten = conv_block(ch_2, ch_2)
#         self.upsample = nn.Upsample(scale_factor=2)
#
#         # spatial attention for F_l
#         self.compress = ChannelPool()
#         self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
#
#         # coor(ch2,r2)(384,4)trans
#         self.poolh = nn.AdaptiveAvgPool2d((None, 1))
#         self.poolw = nn.AdaptiveAvgPool2d((1, None))
#         middle = max(8, ch_1 // r_2)
#         self.conv1 = nn.Conv2d(ch_1, middle, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(middle)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(middle, ch_1, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(middle, ch_1, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#         # bi-linear modelling for both
#         self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (256到256）
#         self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (384，256)
#         self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
#
#         self.dropout = nn.Dropout2d(drop_rate)
#         self.drop_rate = drop_rate
#
#     def forward(self, g, x):
#         # bilinear pooling  g是cnn  x是transformer
#         W_g = self.W_g(g)  # 卷积Conv(ch_1, ch_int, 1, bn=True, relu=False)  (256，256)
#         W_x = self.W_x(x)  # (384,256)
#         bp = self.W(W_g * W_x)  # 乘以（中间的）  (256,256)
#
#         # spatial attention for cnn branch
#         # g_in = g
#         # g = self.compress(g)
#         # g = self.spatial(g)
#         # g = self.sigmoid(g) * g_in
#
#         #cood
#         identity = g
#         batch_size, c, h, w = g.size()  # [batch_size, c, h, w]
#         # X Avg Pool
#         g_h = self.poolh(g)  # [batch_size, c, h, 1]
#
#         # Y Avg Pool
#         g_w = self.poolw(g)  # [batch_size, c, 1, w]
#         g_w = g_w.permute(0, 1, 3, 2)  # [batch_size, c, w, 1]
#
#         # following the paper, cat x_h and x_w in dim = 2，W+H
#         # Concat + Conv2d + BatchNorm + Non-linear
#         y = torch.cat((g_h, g_w), dim=2)  # [batch_size, c, h+w, 1]
#         y = self.act(self.bn1(self.conv1(y)))  # [batch_size, c, h+w, 1]
#         # split
#         g_h, g_w = torch.split(y, [h, w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
#         g_w = g_w.permute(0, 1, 3, 2)  # 把dim=2和dim=3交换一下，也即是[batch_size,c,w,1] -> [batch_size, c, 1, w]
#         # Conv2d + Sigmoid
#         attention_h = self.sigmoid(self.conv_h(g_h))
#         attention_w = self.sigmoid(self.conv_w(g_w))
#         # re-weight
#         g = identity * attention_h * attention_w
#
#         #sa
#         x_res = self.conv(x)
#         yx = self.avg_pool(x)
#         yx = self.conv_atten(yx)
#         yx = self.upsample(yx)
#         x=(yx * x_res) + yx
#
#         # channel attetion for transformer branch   #（x）
#         # x_in = x
#         # x = x.mean((2, 3), keepdim=True)
#         # x = self.fc1(x)
#         # x = self.relu(x)
#         # x = self.fc2(x)
#         # x = self.sigmoid(x) * x_in
#
#         fuse = self.residual(torch.cat([g, x, bp], 1))  # x是 trans分支，g是cnn分支，，bp是中间的
#
#         if self.drop_rate > 0:
#             return self.dropout(fuse)
#         else:
#             return fuse

class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block  ch1是cnn，ch2是transformer   ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256,
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)  # //除完取整
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        #
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_2, ch_2)
        self.conv_atten = conv_block(ch_2, ch_2)
        self.upsample = nn.Upsample(scale_factor=2)

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # coor(ch2,r2)(384,4)trans
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.poolw = nn.AdaptiveAvgPool2d((1, None))
        middle = max(8, ch_1 // r_2)
        self.conv1 = nn.Conv2d(ch_1, middle, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(middle, ch_1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(middle, ch_1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (256到256）
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (384，256)
        self.cnn = add_conv(256, 256, 1, 1)
        self.trans = add_conv(384, 256, 1, 1)
        self.weight1 = add_conv(256, 16, 1, 1)
        self.weight2 = add_conv(256, 16, 1, 1)
        self.weight_trans = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)

        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling  g是cnn，256  x是transformer，384
        W_g = self.cnn(g)  # 卷积Conv(ch_1, ch_int, 1, bn=True, relu=False)  (256，256)
        W_x = self.W_x(x) # (384,256)
        #W_x =self.trans(x)
        # W_x = self.cnn(g)
        #bp = self.W(W_g * W_x)  # 乘以（中间的）  (256,256)
        # fuse=W_g+W_x
        fuse=g


        # #cood
        # identity = g
        # batch_size, c, h, w = g.size()  # [batch_size, c, h, w]
        # # X Avg Pool
        # g_h = self.poolh(g)  # [batch_size, c, h, 1]
        #
        # # Y Avg Pool
        # g_w = self.poolw(g)  # [batch_size, c, 1, w]
        # g_w = g_w.permute(0, 1, 3, 2)  # [batch_size, c, w, 1]
        #
        # # following the paper, cat x_h and x_w in dim = 2，W+H
        # # Concat + Conv2d + BatchNorm + Non-linear
        # y = torch.cat((g_h, g_w), dim=2)  # [batch_size, c, h+w, 1]
        # y = self.act(self.bn1(self.conv1(y)))  # [batch_size, c, h+w, 1]
        # # split
        # g_h, g_w = torch.split(y, [h, w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
        # g_w = g_w.permute(0, 1, 3, 2)  # 把dim=2和dim=3交换一下，也即是[batch_size,c,w,1] -> [batch_size, c, 1, w]
        # # Conv2d + Sigmoid
        # attention_h = self.sigmoid(self.conv_h(g_h))
        # attention_w = self.sigmoid(self.conv_w(g_w))
        # # re-weight
        # g = identity * attention_h * attention_w
        #
        # #sa
        # x_res = self.conv(x)
        # yx = self.avg_pool(x)
        # yx = self.conv_atten(yx)
        # yx = self.upsample(yx)
        # x=(yx * x_res) + yx
        #
        # W_g = self.cnn(g)
        # W_x = self.trans(x)
        #
        # weight1=self.weight1(W_g)
        # weight2=self.weight2(W_x)
        # weight=torch.cat((weight1,weight2),1)
        # weight= self.weight_trans(weight)
        # weight= F.softmax(weight, dim=1)
        # fuse=W_g * weight[:,0:1,:,:]+\
        #      W_x * weight[:,1:2,:,:]

        #fuse = self.residual(torch.cat([g, x], 1))  # x是 trans分支，g是cnn分支，，bp是中间的

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse
class Up(nn.Module):
    """Upscaling then double conv"""   #Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)   #in,out

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

class reconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()


        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)   #in,out

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        #x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)
class Up4(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)   #in,out

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

class down(nn.Module):
    """Upscaling then doudle conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.down =nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)   #in,out

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.down(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

class down4(nn.Module):
    """Upscaling then doudle conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.down =nn.MaxPool2d(4, 4, ceil_mode=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)   #in,out

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.down(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)
class Attention_block(nn.Module):
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
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        x001=g1 + x1
        psi = self.relu(g1 + x1)
        x002=psi
        psi=self.avgpool(psi)

        #psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            xemd1=x
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape    #

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

#@BACKBONES.register_module()
class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=768,
                 classes=1,
                 patch_size=4,  #
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pretrained = False
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.Up1 = up_conv(ch_in=768, ch_out=384)    #unet模块
        self.Up_conv1 = conv_block(ch_in=768, ch_out=384)
        self.Up2 = up_conv(ch_in=384, ch_out=192)
        self.Up_conv2 = conv_block(ch_in=384, ch_out=192)
        self.Up3 = up_conv(ch_in=192, ch_out=96)
        self.Up_conv3 = conv_block(ch_in=192, ch_out=96)
        self.Up_conv4 = conv_block(ch_in=96, ch_out=64)

        self.resnet = resnet()
        if pretrained:
           self.resnet.load_state_dict(torch.load('pretrained/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        #self.resnet.layer4 = nn.Identity()

        # channel attention for F_g, use SE Block  ch1是cnn，ch2是transformer   ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256,
        self.fc1 = nn.Conv2d(64, 64 // 4, kernel_size=1)  # //除完取整
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(64 // 4, 64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # # spatial attention for F_l
        self.sigmoid = nn.Sigmoid()
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

       # self.transformer = deit(pretrained=pretrained)

        self.up1 = Up(in_ch1=384, out_ch=128)  #
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(  # （256，64）（64，64） （64，numclass）
            Conv(256, 64, 1, bn=True, relu=True),  # （64，64） （64，numclass）
            Conv(64, 64, 3, bn=True, relu=True),  # （64，64） （64，numclass）
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )
        #self.sa64=SqueezeAttentionBlock(ch_in=64, ch_out=64)
        #self.sa128 = SqueezeAttentionBlock(ch_in=128, ch_out=128)
        #self.sa256 = SqueezeAttentionBlock(ch_in=256, ch_out=256)
        # self.adam=adam(ch_in=3)
        # self.adam1 = adam(ch_in=64)
        # self.adam2 = adam(ch_in=128)
        # self.adam3 = adam(ch_in=256)
        # self.se1 = se_block(channel=256)
        # self.adamt = adam(ch_in=384)
        self.up_c_0 = BiFusion_block(ch_1=512, ch_2=512, r_2=8, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)  #
        self.up_c_0_2 = Up(in_ch1=512, out_ch=256, in_ch2=256, attn=True)

        self.up_c = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)  #

        #self.up_c_1_1 = Up(256, 128) #
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=False)

        #self.up_c_2_1 = Up(128, 64)
        self.up_c_2_2 = Up(128, 64, 64, attn=False)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=256, out_size=22, scale_factor=(48, 64))
        self.dsv3 = UnetDsv3(in_size=128, out_size=22, scale_factor=(48, 64))
        self.dsv2 = UnetDsv3(in_size=64, out_size=22, scale_factor=(48, 64))


        self.scale_att = scale_atten_convblock(in_size=64, out_size=64)

        self.drop = nn.Dropout2d(drop_rate)

        self.b1tob3 = down4(96, 96)#nn.MaxPool2d(4, 4, ceil_mode=True)     #7.5  多尺度融合修改    #加两层卷积  加残差映射
        # self.b1tob3_conv = nn.Conv2d(96, 96, 3, padding=1)  # 64,64
        # self.b1tob3_bn = nn.BatchNorm2d(96)
        # self.b1tob3_relu = nn.ReLU(inplace=True)

        self.b2tob3 =down(192, 96) #nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.b2tob3_conv = nn.Conv2d(192, 96, 3, padding=1)  # 64,64
        # self.b2tob3_bn = nn.BatchNorm2d(96)
        # self.b2tob3_relu = nn.ReLU(inplace=True)

        self.b3tob3_conv =reconv(384,96)#nn.Conv2d(384, 96, 3, padding=1)  # 64,64
        # self.b3tob3_bn = nn.BatchNorm2d(96)
        # self.b3tob3_relu = nn.ReLU(inplace=True)

        self.b4tob3 =Up(768, 96) #Up(128, 64)  #nn.Upsample(scale_factor=2, mode='bilinear')
        #self.b4tob3_conv = nn.Conv2d(768, 96, 3, padding=1)  # 64,64
        #self.b4tob3_bn = nn.BatchNorm2d(96)
        #self.b4tob3_relu = nn.ReLU(inplace=True)

        self.b3_conv = reconv(384,384)#nn.Conv2d(384, 384, 3, padding=1)  # 320,320
        # self.b3_bn = nn.BatchNorm2d(384)
        # self.b3_relu = nn.ReLU(inplace=True)

        self.xc2toxc1 = down(64, 42)#nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.xc2toxc1_conv = nn.Conv2d(64, 42, 3, padding=1)  # 64,64
        # self.xc2toxc1_bn = nn.BatchNorm2d(42)
        # self.xc2toxc1_relu = nn.ReLU(inplace=True)


        self.xc1toxc1_conv = nn.Conv2d(128, 44, 3, padding=1)  # 64,64
        self.xc1toxc1_bn = nn.BatchNorm2d(44)
        self.xc1toxc1_relu = nn.ReLU(inplace=True)

        self.xctoxc1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.xctoxc1_conv = nn.Conv2d(256, 42, 3, padding=1)  # 64,64
        self.xctoxc1_bn = nn.BatchNorm2d(42)
        self.xctoxc1_relu = nn.ReLU(inplace=True)

        self.xc1_conv = nn.Conv2d(128, 128, 3, padding=1)  # 320,320
        self.xc1_bn = nn.BatchNorm2d(128)
        self.xc1_relu = nn.ReLU(inplace=True)

        self.xb2_conv = reconv(96, 64)
        self.xc0_conv = reconv(768, 512)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(384, 384)
        # self.conv_res=nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_atten = conv_block(384, 384)
        self.upsample = nn.Upsample(scale_factor=2)

        self.atrous_block1 = nn.Conv2d(96, 96, 1, 1)
        self.atrous_block6 = nn.Conv2d(96, 96, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(96, 96, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(96, 96, 3, 1, padding=18, dilation=18)
        self.conv3 = nn.Conv2d(22, 22, 3, 1, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(22)
        #nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv4 = nn.Conv2d(22, 22, 3, 1, padding=12, dilation=12)
        self.bn4 = nn.BatchNorm2d(22)
        self.conv2 = conv3x3(22, 20)
        self.bn2 = nn.BatchNorm2d(20)
        self.sa=sa_layer(channel=64)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        d1=x
        dd1=self.pretrain_img_size
        #x = self.adam(x)
        x = self.patch_embed(x)
        dd2=x
        #x=sa_layer(x)


        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)            #x=2，3136，96【0.5657，，0.8514...】

        outs = []
        for i in range(self.num_layers):

            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:                  #if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                s1=self.num_features[i]
                s2 = self.num_features #96,192,384,768
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        a4=outs
        b4=a4[3]   #768.7.7             #
        b3=a4[2]   #384
        b2=a4[1]   #192
        b1=a4[0]   #96

        #b4 = sa_layer(b4)
        x_b = b3
        # x_b = self.adamt(x_b)

        #x_b=self.drop(x_b)

        # x_0, x_1, x_2,x_3= x_b.chunk(4, dim=1)  #192
        # x_0 = self.atrous_block1(x_0)
        # x_1 = self.atrous_block6(x_1)
        # x_2 = self.atrous_block12(x_2)
        # x_3 = self.atrous_block18(x_3)
        #
        # x_0_in = x_0
        # x_0 = self.compress(x_0)
        # m3= x_0
        # x_0 = self.spatial(x_0)
        # m4=x_0
        # x_0 = self.sigmoid(x_0) * x_0_in
        #
        # x_1_in = x_1
        # x_1= self.compress(x_1)
        # x_1 = self.spatial(x_1)
        # x_1 = self.sigmoid(x_1) * x_1_in
        #
        # x_2_in = x_2
        # x_2 = self.compress(x_2)
        # x_2 = self.spatial(x_2)
        # x_2 = self.sigmoid(x_2) * x_2_in
        #
        # x_3_in = x_3
        # x_3 = self.compress(x_3)
        # x_3 = self.spatial(x_3)
        # x_3 = self.sigmoid(x_3) * x_3_in
        #
        # x_b= torch.cat([x_0, x_1, x_2,x_3], dim=1)
        # top-down path    （192，256）
        x_u = self.resnet.conv1(d1)  # (通道由3变64，大小变为1/2)     #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        e30 = x_u
        x_u = self.resnet.bn1(x_u)
        e31 = x_u
        x_u = self.resnet.relu(x_u)
        e32 = x_u
        x_u = self.resnet.maxpool(x_u)  # 大小变为1/2（初始的1/4/）
        e4 = x_u

        x_u_2 = self.resnet.layer1(x_u)  # self.layer1 = self._make_layer(block, 64, layers[0]
        e5 = x_u_2
        x_u_2 = self.drop(x_u_2)
        e6 = x_u_2
        #x_u_2 = self.adam1(x_u_2)

        x_u_1 = self.resnet.layer2(x_u_2)  # 通道加倍，高宽减半（初始的1/8）
        e7 = x_u_1
        x_u_1 = self.drop(x_u_1)
        e8 = x_u_1
        #x_u_1 = self.adam2(x_u_1)

        x_u = self.resnet.layer3(x_u_1)  # 通道加倍，高宽减半（初始的1/16）
        e9 = x_u
        x_u = self.drop(x_u)
        e10 = x_u
        #x_u=self.adam3(x_u)


        #e1000=self.resnet.layer4
        # x_u_0=self.resnet.layer4(x_u)  # 通道加倍，高宽减半（初始的1/16）
        # e101 = x_u_0
        # x_u_0 = self.drop(x_u_0)      #1/32
        # e102 = x_u_0

        #x_c_0_0=self.up_c_0(x_u_0, x_b_0)  #1/32

        # joint path
        x_c = self.up_c(x_u, x_b)  # BiFision        xb应该是1/16(trans)      xu cnn    #256

        x_c_1 = self.up_c_1_2(x_c, x_u_1)

        x_c_2 = self.up_c_2_2(x_c_1, x_u_2)

        # Deep Supervision
        dsv4 = self.dsv4(x_c)  # up4:128,1/8  #256-22,1/16
        dsv3 = self.dsv3(x_c_1)  # up3:64,1/4   #128-21,1/8
        dsv2 = self.dsv2(x_c_2)  # up2:32,1/2   #64-22,1/4

        ds4=self.relu(self.bn4(self.conv4(dsv4)))
        ds3=self.relu(self.bn3(self.conv3(ds4+dsv3)))
        ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        ds4 = self.drop(ds4)
        ds3= self.drop(ds3)
        ds2 = self.drop(ds2)
        dsv_cat = torch.cat([ds2, ds3, ds4], dim=1)
        #dsv_cat = self.drop(dsv_cat)
        #out = self.scale_att(dsv_cat)
        #out = self.se1(out)
        #dsv_cat=self.sa(dsv_cat)

        # decoder part

        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear')  # 最后融合


        #return b1   #return tuple(outs)
        return map_2

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

if __name__ == '__main__':
    # input = torch.rand(2, 3, 224, 320)
    # # model = RCEM(3, 64)
    # # model = BasicBlock(3, 64, 64)
    # model = CE_Net(classes=1)
    # out12 = model(input)
    # print(out12.shape)
    model = SwinTransformer()
    # model.load_state_dict(torch.load('C:/Users/89444/Desktop/TransFuse-2/snapshots/U-net/TransFuse-1.pth'))

    # 现在假设你已经准备好训练好的模型和预处理输入了
    class_num = 1

    model_ft = model()
    model_ft.load_state_dict(torch.load('snapshots/U-net/2018swinFuse9.166-1.pth',
                                        map_location=lambda storage, loc: storage))