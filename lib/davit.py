import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.parameter import Parameter
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
# from torchsummary import summary
# import torchinfo
from torchvision.models import resnet34 as resnet
#from Models.layers.modules import UnetDsv3
from torchvision.models import resnet50
from torchvision import models
from lib import UCTransNet as gsc
# from lib.GSnorm import Norm2d
from torchvision import transforms
import numpy as np
import cv2
plains = [16, 32, 64, 128, 256]
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
#DeformConv2d(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False)

# class attnUp(nn.Module):
#     """Upscaling then double conv"""
#
#     def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
#         super().__init__()
#
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)
#
#         if attn:
#             self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
#         else:
#             self.attn_block = None
#
#     def forward(self, x1, x2=None):
#
#         x1 = self.up(x1)
#         # input is CHW
#         if x2 is not None:
#             diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#             diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
#
#             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                             diffY // 2, diffY - diffY // 2])
#
#             if self.attn_block is not None:
#                 x2 = self.attn_block(x1, x2)
#             x1 = torch.cat([x2, x1], dim=1)
#         x = x1
#         return self.conv(x)

# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#SqueezeAttentionBlock(ch_in, ch_out)
class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # print(x.shape)
        x_res = self.conv(x)
        x1=x_res
        # print(x_res.shape)
        y = self.avg_pool(x)
        x2=y
        # print(y.shape)
        y = self.conv_atten(y)
        x3=y
        # print(y.shape)
        y = self.upsample(y)
        # x4=y
        # # print(y.shape, x_res.shape)
        # x5=y * x_res
        # x6=x5+y

        return (y * x_res) + y
class MPA(nn.Module):
    def __init__(self, filters):
        super(MPA, self).__init__()
        self.levels = nn.ModuleList([self._conv_1x1(f_in, filters[0]) for f_in in filters[1:]])
        self.conv_out = nn.Sequential(
            nn.Conv2d(filters[0] * len(filters), filters[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            SEBlock(filters[0]),
        )

    @classmethod
    def _conv_1x1(cls, in_channels, out_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        return nn.Sequential(conv, bn)

    def forward(self, features):
        feature_list = [self.levels[i](f) for i, f in enumerate(features[1:])]
        _, _, h, w = features[0].size()
        feature_outs = [F.upsample(input=f, size=(h, w), mode='bilinear') for f in feature_list]
        feature_outs = [features[0]] + feature_outs
        feature_outs = torch.cat(feature_outs, dim=1)
        return self.conv_out(feature_outs)
# Attention gate代码

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze  [N, C, 1, 1] -->view [N, C]
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y
nonlinearity = partial(F.relu, inplace=True)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        #self.conv1 = ODConv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        #ODConv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        #self.conv3 = ODConv2d(in_channels // 4, n_filters,  kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(in_channels, in_channels // 4)
        self.conv_atten = conv_block(in_channels, in_channels // 4)
        self.upsample = nn.Upsample(scale_factor=2)
        #self.upsample=F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)

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


class DecoderBlock33(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock33, self).__init__()
        #self.conv1 = ODConv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        #ODConv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=4, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        #self.conv3 = ODConv2d(in_channels // 4, n_filters,  kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(in_channels // 4, in_channels // 4)
        self.conv_atten = conv_block(in_channels // 4, in_channels // 4)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        # x_res = self.conv(x)
        # x_res=x
        # y = self.avg_pool(x)
        # x2 = y
        # y = self.conv_atten(y)
        # x3 = y
        # y = self.upsample(y)
        # x = (y * x_res) + y
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x





class Att_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Att_block, self).__init__()
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
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        x001 = g1 + x1
        psi = self.relu(g1 + x1)
        x002 = psi
        psi = self.avgpool(psi)

        # psi = self.psi(psi)
        return x * psi



class spAttention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(spAttention_block, self).__init__()
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
        # g1 = self.W_g(g)
        # #a=x
        # x1 = self.W_x(x)
        # #b=g1+x1
        #
        # psi = self.relu(g1+x1)
        # psi = self.psi(psi)

        g1 = self.W_g(g)
        # a=x
        x1 = self.W_x(x)
        # b=g1+x1
        f1 = g1 + x1
        f1_in = f1
        f1 = f1.mean((2, 3), keepdim=True)
        f1 = self.fc1(f1)
        f1 = self.relu(f1)
        f1 = self.fc2(f1)
        f1 = self.sigmoid(f1) * f1_in
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x



def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)
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

# class Up(nn.Module):  #2022118
#     """Upscaling then double conv"""   #Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)self.up_c_1(x_c, x_c1)
#
#     def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
#         super().__init__()
#
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)   #in,out
#         self.fam=FAMBlock(channels=out_ch)
#
#     def forward(self, x1, x2):
#
#         x1 = self.up(x1)
#         # input is CHW
#         if x2 is not None:
#             # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#             # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
#             #
#             # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#             #                 diffY // 2, diffY - diffY // 2])
#             x1 = torch.cat([x2, x1], dim=1)
#         x = x1
#         return self.conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)
        self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)


    def forward(self, x1, x2):
   #self.up_c_1(x_c, x_u_1)
        A1=x1
        a2=x2
        x1 = self.up(x1)
        a3=x1
        x2 = self.attn_block(x1, x2)
        x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)
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



class h_swish(nn.Module):
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x
class BiFusion_block(nn.Module):
    def  __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()
        # coor(ch2,r2)(384,4)trans
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=True)  # (256到256）
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=True)  # (384，256)
        self.W_f = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        #
        # self.W_g=odconv1x1(ch_1, ch_int, reduction=0.0625, kernel_num=1)
        # self.W_x = odconv1x1(ch_2, ch_int, reduction=0.0625, kernel_num=1)

        self.relu = nn.ReLU(inplace=True)
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.inplanes = ch_1
        self.Conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1)
        r = 8
        L = 48
        d = max(int(self.inplanes / r), L)
        self.W1 = nn.Linear(2 * self.inplanes, d)
        self.W2 = nn.Linear(d, self.inplanes)
        self.W3 = nn.Linear(d, self.inplanes)

    def forward(self, g, x):

        U1 = self.W_g(g)
        U2 = self.W_x(x)
        # W_g = self.W_g(g)
        # W_x = self.W_x(x)

        bp = self.W_f(U1 * U2)

        #fuse = g + x
        U=U1+U2

        avg_pool = F.avg_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        max_pool = F.max_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        # p = torch.mean(torch.mean(U, dim=2), dim=2).reshape(-1, self.inplanes, 1, 1)
        # q = torch.matmul(
        #     U.reshape([-1, self.inplanes, x.shape[2] * x.shape[3]]),
        #     (nn.Softmax(dim=1)(self.Conv_q(x))).reshape([-1, x.shape[2] * x.shape[3], 1])
        # ).reshape([-1, self.inplanes, 1, 1])
        sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
        # s = torch.sigmoid(torch.cat([p, q], 1)).reshape([-1, self.inplanes * 2])
        # z1 = self.W2(nn.ReLU()(self.W1(s)))
        # z2 = self.W3(nn.ReLU()(self.W1(s)))
        zc1 = self.W2(nn.ReLU()(self.W1(sc)))
        zc2 = self.W3(nn.ReLU()(self.W1(sc)))
        # a1 = (torch.exp(z1) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
        # a2 = (torch.exp(z2) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
        ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        fuse = U1 * ac1 + U2 * ac2+U2


        #fuse = self.residual(torch.cat([g, x], 1))  # x是 trans分支，g是cnn分支，，bp是中间的
        return fuse


import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# from mmcv_custom import load_checkpoint
# from mmdet.utils import get_root_logger
# from ..builder import BACKBONES


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=False):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=96,
            overlapped=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        if patch_size[0] == 4:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(7, 7),
                stride=patch_size,
                padding=(3, 3))
            self.norm = nn.LayerNorm(embed_dim)
        if patch_size[0] == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=to_2tuple(kernel),
                stride=patch_size,
                padding=to_2tuple(pad))
            self.norm = nn.LayerNorm(in_chans)

    def forward(self, x, size):
        H, W = size
        dim = len(x.shape)
        input = torch.rand(2, 3, 224, 320)

        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B,
                          H,
                          W,
                          C).permute(0, 3, 1, 2).contiguous()

        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        if dim == 4:
            x = self.norm(x)
        return x, newsize


class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


def window_partition(x, window_size: int):
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


def window_reverse(windows, window_size: int, H: int, W: int):
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
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SpatialBlock(nn.Module):
    r""" Windows Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


# @BACKBONES.register_module()
class DaViT(nn.Module):
    r""" Dual Attention Transformer

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, in_chans=3, classes=1, channels=3,depths=(1, 1, 3, 1), patch_size=4,
                 embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_path_rate=0.1, norm_layer=nn.LayerNorm, attention_types=('spatial', 'channel'),
                 ffn=True, overlapped_patch=False, cpe_act=False, weight_init='',
                 drop_rate=0., attn_drop_rate=0., img_size=224
                 ):
        super().__init__()

        self.num_classes = classes
        architecture = [[index] * item for index, item in enumerate(depths)]
        self.architecture = architecture
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_stages = len(self.embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*self.architecture))))]
        assert self.num_stages == len(self.num_heads) == (sorted(list(itertools.chain(*self.architecture)))[-1] + 1)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        #conresnet=OD_ResNet(BasicBlock, [3, 4, 6, 3])
        #conresnet=convres
        self.firstconv=resnet.conv1
        # self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        # self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        # self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        # self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.up_c_1 = Up(in_ch1=512, out_ch=256, in_ch2=256)
        # self.up_c_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
        # self.up_c_3 = Up(in_ch1=128, out_ch=64, in_ch2=64)



        # self.Attentiongate1 = AttentionBlock(256, 256, 256)
        # self.Attentiongate2 = AttentionBlock(128, 128, 128)
        # self.Attentiongate3 = AttentionBlock(64, 64, 64)

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=768, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=192, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=96, r_2=4, ch_int=64, ch_out=64)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

        self.finalconv4 = nn.Conv2d(64, classes, 3, padding=1)
        # self.se=SEBlock(channel=512)
        self.sa4=SqueezeAttentionBlock(ch_in=512, ch_out=512)
        self.sa3 = SqueezeAttentionBlock(ch_in=256, ch_out=256)
        self.sa2 = SqueezeAttentionBlock(ch_in=128, ch_out=128)
        self.sa1 = SqueezeAttentionBlock(ch_in=64, ch_out=64)

        # deep supervision
        #Shape Stream
        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = BasicBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = BasicBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = BasicBlock(8, 8)
        self.d3 = nn.Conv2d(8, 4, kernel_size=1)
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)



        self.expand = nn.Sequential(nn.Conv2d(1, 32, kernel_size=1),
                                    Norm2d(32),
                                    nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

        self.norm00 = nn.LayerNorm(96)
        self.norm01 = nn.LayerNorm(192)
        self.norm02 = nn.LayerNorm(384)
        self.norm03 = nn.LayerNorm(768)



        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=patch_size if i == 0 else 2,
                       in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                       embed_dim=self.embed_dims[i],
                       overlapped=overlapped_patch)
            for i in range(self.num_stages)])

        main_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))

            block = nn.ModuleList([
                MySequential(*[
                    ChannelBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        cpe_act=cpe_act
                    ) if attention_type == 'channel' else
                    SpatialBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        cpe_act=cpe_act,
                        window_size=window_size,
                    ) if attention_type == 'spatial' else None
                    for attention_id, attention_type in enumerate(attention_types)]
                             ) for layer_id, item in enumerate(block_param)
            ])
            main_blocks.append(block)
        self.main_blocks = nn.ModuleList(main_blocks)

        # add a norm layer for each output
        for i_layer in range(self.num_stages):
            layer = norm_layer(self.embed_dims[i_layer])  # if i_layer != 0 else nn.Identity()
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     logger = get_root_logger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        c1=x
        x, size = self.patch_embeds[0](x, (x.size(2), x.size(3)))
        x_out1=[x]
        outs = [x]
        features = [x]
        sizes = [size]
        branches = [0]
        depths = (1, 1, 3, 1)
        ar = [[index] * item for index, item in enumerate(depths)]
        C = 0

        for block_index, block_param in enumerate(self.architecture):
            # ab=block_index
            # print(ab)
            # abl=block_param
            # print(abl)
            aa1=x
            branch_ids = sorted(set(block_param))
            for branch_id in branch_ids:   #0 1 2 3
                if branch_id not in branches:
                    aem=x
                    aee=size
                    x, size = self.patch_embeds[branch_id](features[-1], sizes[-1])
                    aem1 = x
                    aee1= size
                    features.append(x)
                    sizes.append(size)
                    branches.append(branch_id)
            for layer_index, branch_id in enumerate(block_param):  #branch_id：012223
                a=branch_id
                print(layer_index,a,"_______")
                features[branch_id], _ = self.main_blocks[block_index][layer_index](features[branch_id],
                                                                                    sizes[branch_id])
                if branch_id==0:
                    features[branch_id]=self.norm00(features[branch_id])
                    H, W = sizes[0]

                    out= features[branch_id].view(-1, H, W, 96).permute(0, 3, 1, 2).contiguous()

                    #features[0]=outs[0]
                if branch_id==1:
                    features[branch_id]=self.norm01(features[branch_id])
                if branch_id==2:
                    C=C+1
                    print(C,"dddd-=-=-")
                if C == 2:
                    features[branch_id] = self.norm02(features[branch_id])
                if branch_id==3:
                    features[branch_id]=self.norm03(features[branch_id])
            # for i in range(self.num_stages):
            #     print(i, "iiiiii")
            #     norm_layer = getattr(self, f'norm{i}')
            #     x_out = norm_layer(features[i])
            #     H, W = sizes[i]
            #     out = x_out.view(-1, H, W, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
            #
            #     outs.append(out)

                #norm_layer = getattr(self, f'norm{i}')
                #x_out1[branch_id] = norm_layer(features[i])
                #x_out2[branch_id] = norm_layer(features[i])
                # features[branch_id]=x_out
                # H, W = sizes[i]
                # i=1
                # a=i
                # i=a+1
                # fe=features

        outs = []
        # for i in range(self.num_stages):
        #     print(i, "iiiiii")
        #     norm_layer = getattr(self, f'norm{i}')
        #     x_out = norm_layer(features[i])
        #     H, W = sizes[i]
        #     out = x_out.view(-1, H, W, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
        #     outs.append(out)
        x_t=outs[3]
        x_t_1=outs[2]
        x_t_2=outs[1]  #192
        x_t_3=outs[0]  #96
        x_v_t=x_t_3
        x=c1
        x = self.firstconv(x)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(x)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]


        x_c = self.fusion_c(x_u, x_t)  # 512,1/32
        x_c_1 = self.fusion_c1(x_u_1, x_t_1)  # 256
        x_c_2 = self.fusion_c2(x_u_2, x_t_2)  # 128
        x_c_3 = self.fusion_c3(x_u_3, x_t_3)  # 64
        x_v_f = x_c_3

        d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
        d1 = self.decoder1(d2)  # [2, 64, 112, 160]


        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        xv=out
        out = self.finalrelu1(out)
        xv1 = out
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        xv2 = out
        out = self.finalrelu2(out)

        # out = self.finalconv3(out)  # [2, 2, 224, 320]
        out = self.finalconv4(out)


        # x_v_d1 = F.interpolate(d1, size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d2 = F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d3 = F.interpolate(d3, size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d1 = x_v_d1.view(224, 224)
        # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        # pic = toPIL(x_v_d1)
        # pic.save('E:/fenge/TransFuse-1/data/random3.jpg')
        # x_v_d4 = F.interpolate(d4, size=(224, 224), mode='bilinear', align_corners=False)

        return out

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DaViT, self).train(mode)

# class DaViT(nn.Module):
#     r""" Dual Attention Transformer
#
#     Args:
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
#         num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#     """
#
#     def __init__(self, in_chans=3, classes=1, channels=3,depths=(1, 1, 3, 1), patch_size=4,
#                  embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
#                  qkv_bias=True, drop_path_rate=0.1, norm_layer=nn.LayerNorm, attention_types=('spatial', 'channel'),
#                  ffn=True, overlapped_patch=False, cpe_act=False, weight_init='',
#                  drop_rate=0., attn_drop_rate=0., img_size=224
#                  ):
#         super().__init__()
#
#         self.num_classes = classes
#         architecture = [[index] * item for index, item in enumerate(depths)]
#         self.architecture = architecture
#         self.embed_dims = embed_dims
#         self.num_heads = num_heads
#         self.num_stages = len(self.embed_dims)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*self.architecture))))]
#         assert self.num_stages == len(self.num_heads) == (sorted(list(itertools.chain(*self.architecture)))[-1] + 1)
#
#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#         #conresnet=OD_ResNet(BasicBlock, [3, 4, 6, 3])
#         #conresnet=convres
#         self.firstconv=resnet.conv1
#         # self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         # self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         # self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         # self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#         # self.encoder1 = resnet.layer1
#         # self.encoder2 = resnet.layer2
#         # self.encoder3 = resnet.layer3
#         # self.encoder4 = resnet.layer4
#         self.decoder4 = DecoderBlock(512, filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#         # self.up_c_1 = Up(in_ch1=512, out_ch=256, in_ch2=256)
#         # self.up_c_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
#         # self.up_c_3 = Up(in_ch1=128, out_ch=64, in_ch2=64)
#
#
#
#         # self.Attentiongate1 = AttentionBlock(256, 256, 256)
#         # self.Attentiongate2 = AttentionBlock(128, 128, 128)
#         # self.Attentiongate3 = AttentionBlock(64, 64, 64)
#
#         self.fusion_c = BiFusion_block(ch_1=512, ch_2=768, r_2=4, ch_int=512, ch_out=512)
#         self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256)
#         self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=192, r_2=4, ch_int=128, ch_out=128)
#         self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=96, r_2=4, ch_int=64, ch_out=64)
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)
#         # self.se=SEBlock(channel=512)
#         self.sa4=SqueezeAttentionBlock(ch_in=512, ch_out=512)
#         self.sa3 = SqueezeAttentionBlock(ch_in=256, ch_out=256)
#         self.sa2 = SqueezeAttentionBlock(ch_in=128, ch_out=128)
#         self.sa1 = SqueezeAttentionBlock(ch_in=64, ch_out=64)
#
#         # deep supervision
#         self.final_3 = nn.Sequential(
#             Conv(512, 64, 3, bn=True, relu=True),
#             Conv(64, 1, 3, bn=False, relu=False)
#         )
#         self.final_4 = nn.Sequential(
#             Conv(768, 64, 3, bn=True, relu=True),
#             Conv(64, 1, 3, bn=False, relu=False)
#         )
#
#
#         self.patch_embeds = nn.ModuleList([
#             PatchEmbed(patch_size=patch_size if i == 0 else 2,
#                        in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
#                        embed_dim=self.embed_dims[i],
#                        overlapped=overlapped_patch)
#             for i in range(self.num_stages)])
#
#         main_blocks = []
#         for block_id, block_param in enumerate(self.architecture):
#             layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))
#
#             block = nn.ModuleList([
#                 MySequential(*[
#                     ChannelBlock(
#                         dim=self.embed_dims[item],
#                         num_heads=self.num_heads[item],
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias,
#                         drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
#                         norm_layer=nn.LayerNorm,
#                         ffn=ffn,
#                         cpe_act=cpe_act
#                     ) if attention_type == 'channel' else
#                     SpatialBlock(
#                         dim=self.embed_dims[item],
#                         num_heads=self.num_heads[item],
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias,
#                         drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
#                         norm_layer=nn.LayerNorm,
#                         ffn=ffn,
#                         cpe_act=cpe_act,
#                         window_size=window_size,
#                     ) if attention_type == 'spatial' else None
#                     for attention_id, attention_type in enumerate(attention_types)]
#                              ) for layer_id, item in enumerate(block_param)
#             ])
#             main_blocks.append(block)
#         self.main_blocks = nn.ModuleList(main_blocks)
#
#         # add a norm layer for each output
#         for i_layer in range(self.num_stages):
#             layer = norm_layer(self.embed_dims[i_layer])  # if i_layer != 0 else nn.Identity()
#             layer_name = f'norm{i_layer}'
#             self.add_module(layer_name, layer)
#
#     def init_weights(self, pretrained=None):
#         """Initialize the weights in backbone.
#
#         Args:
#             pretrained (str, optional): Path to pre-trained weights.
#                 Defaults to None.
#         """
#
#         def _init_weights(m):
#             if isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=.02)
#                 if isinstance(m, nn.Linear) and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)
#
#         # if isinstance(pretrained, str):
#         #     self.apply(_init_weights)
#         #     logger = get_root_logger()
#         #     load_checkpoint(self, pretrained, strict=False, logger=logger)
#         # elif pretrained is None:
#         #     self.apply(_init_weights)
#         # else:
#         #     raise TypeError('pretrained must be a str or None')
#
#     def forward(self, x):
#         c1=x
#         x, size = self.patch_embeds[0](x, (x.size(2), x.size(3)))
#         features = [x]
#         sizes = [size]
#         branches = [0]
#
#         for block_index, block_param in enumerate(self.architecture):
#             branch_ids = sorted(set(block_param))
#             for branch_id in branch_ids:
#                 if branch_id not in branches:
#                     x, size = self.patch_embeds[branch_id](features[-1], sizes[-1])
#                     features.append(x)
#                     sizes.append(size)
#                     branches.append(branch_id)
#             for layer_index, branch_id in enumerate(block_param):
#                 features[branch_id], _ = self.main_blocks[block_index][layer_index](features[branch_id],
#                                                                                     sizes[branch_id])
#
#         outs = []
#         for i in range(self.num_stages):
#             norm_layer = getattr(self, f'norm{i}')
#             x_out = norm_layer(features[i])
#             H, W = sizes[i]
#             out = x_out.view(-1, H, W, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#         x_t=outs[3]
#         x_t_1=outs[2]
#         x_t_2=outs[1]  #192
#         x_t_3=outs[0]  #96
#         x_v_t=x_t_3
#         x=c1
#         x = self.firstconv(x)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#
#         x = self.firstmaxpool(x)  # [2, 64, 56, 80]
#         x_u_3 = self.encoder1(x)  # [2, 64, 56, 80]
#         x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
#         x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
#         x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]
#         x_v_c=x_u_3
#
#         x_c = self.fusion_c(x_u, x_t)  # 512,1/32
#
#         #x_c= self.sa4(x_c)
#         x_c_1 = self.fusion_c1(x_u_1, x_t_1)  # 256
#         # x_c_1= self.sa3(x_c_1)
#         x_c_2 = self.fusion_c2(x_u_2, x_t_2)  # 128
#         # x_c_2 = self.sa2(x_c_2)
#         x_c_3 = self.fusion_c3(x_u_3, x_t_3)  # 64
#         # x_c_3 = self.sa1(x_c_3)
#         x_v_f = x_c_3
#
#         d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
#         d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
#         d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
#         d1 = self.decoder1(d2)  # [2, 64, 112, 160]
#         # d4 = self.decoder4(x_u) + x_u_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
#         # d3 = self.decoder3(d4) + x_u_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
#         # d2 = self.decoder2(d3) + x_u_3  # [2, 64, 56, 80]
#         # d1 = self.decoder1(d2)  # [2, 64, 112, 160]
#
#         out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
#         xv=out
#         out = self.finalrelu1(out)
#         xv1 = out
#         out = self.finalconv2(out)  # [2, 32, 224, 320]
#         xv2 = out
#         out = self.finalrelu2(out)
#         xv3 = out
#         out = self.finalconv3(out)  # [2, 2, 224, 320]
#         xv4 = out
#
#         x_v_u = F.interpolate(self.final_3(x_u), scale_factor=32, mode='bilinear', align_corners=True)
#         x_v_t = F.interpolate(self.final_4(x_t), scale_factor=32, mode='bilinear', align_corners=True)
#         x_v_d3 = F.interpolate(d3, size=(224, 224), mode='bilinear', align_corners=False)
#         x_v_d4 = F.interpolate(d4, size=(224, 224), mode='bilinear', align_corners=False)
#
#         return x_v_u,x_v_t,out
#
#     def train(self, mode=True):
#         """Convert the model into training mode while keep layers freezed."""
#         super(DaViT, self).train(mode)

class Unet(nn.Module):
    r""" Dual Attention Transformer

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, in_chans=3, classes=1, channels=3,depths=(1, 1, 3, 1), patch_size=4,
                 embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_path_rate=0.1, norm_layer=nn.LayerNorm, attention_types=('spatial', 'channel'),
                 ffn=True, overlapped_patch=False, cpe_act=False, weight_init='',
                 drop_rate=0., attn_drop_rate=0., img_size=224
                 ):
        super().__init__()

        self.num_classes = classes
        architecture = [[index] * item for index, item in enumerate(depths)]
        self.architecture = architecture
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_stages = len(self.embed_dims)


        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        #conresnet=OD_ResNet(BasicBlock, [3, 4, 6, 3])
        #conresnet=convres
        self.firstconv=resnet.conv1
        # self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        # self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        # self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        # self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])




        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

        # self.finaldeconv1d3 = nn.ConvTranspose2d(128, 64, 4, 8, 1,1)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2d3 = nn.Conv2d(64, 64, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3d3 = nn.Conv2d(64, classes, 3, padding=1)

        # self.cond1=nn.Conv2d(64, classes, 3, padding=1)
        # self.cond2 = nn.Conv2d(64, classes, 3, padding=1)
        # self.cond3 = nn.Conv2d(128, classes, 3, padding=1)
        # self.cond4 = nn.Conv2d(256, classes, 3, padding=1)
        # self.se=SEBlock(channel=512)


        # deep supervision



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


    def forward(self, x):
        x = self.firstconv(x)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)

        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(x)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]


        d4 = self.decoder4(x_u) + x_u_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d3 = self.decoder3(d4) + x_u_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d2 = self.decoder2(d3) + x_u_3  # [2, 64, 56, 80]
        d1 = self.decoder1(d2)  # [2, 64, 112, 160]
        x_v_d1=d1

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        x_v_d0 = out    #256
        x_v_d1= x_v_d0
        x_v_d2=x_v_d0


        x_v_d3 = x_v_d0   #128
        x_v_d4 = x_v_d0
        x_v_d5 = x_v_d0

        x_v_d6 = x_v_d0    #64
        x_v_d7 = x_v_d0
        x_v_d8 =x_v_d0

        return out,x_v_d0,x_v_d1,x_v_d2,x_v_d3,x_v_d4,x_v_d5,x_v_d6,x_v_d7,x_v_d8



class ViT(nn.Module):
    r""" Dual Attention Transformer

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, in_chans=3, classes=1, channels=3,depths=(1, 1, 3, 1), patch_size=4,
                 embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_path_rate=0.1, norm_layer=nn.LayerNorm, attention_types=('spatial', 'channel'),
                 ffn=True, overlapped_patch=False, cpe_act=False, weight_init='',
                 drop_rate=0., attn_drop_rate=0., img_size=224
                 ):
        super().__init__()

        self.num_classes = classes
        architecture = [[index] * item for index, item in enumerate(depths)]
        self.architecture = architecture
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_stages = len(self.embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*self.architecture))))]
        assert self.num_stages == len(self.num_heads) == (sorted(list(itertools.chain(*self.architecture)))[-1] + 1)

        filters = [96, 192, 384, 768]
        resnet = models.resnet34(pretrained=True)
        #conresnet=OD_ResNet(BasicBlock, [3, 4, 6, 3])
        #conresnet=convres
        self.firstconv=resnet.conv1
        # self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        # self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        # self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        # self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(768, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.up_c_1 = Up(in_ch1=512, out_ch=256, in_ch2=256)
        # self.up_c_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
        # self.up_c_3 = Up(in_ch1=128, out_ch=64, in_ch2=64)



        # self.Attentiongate1 = AttentionBlock(256, 256, 256)
        # self.Attentiongate2 = AttentionBlock(128, 128, 128)
        # self.Attentiongate3 = AttentionBlock(64, 64, 64)

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=768, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=192, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=96, r_2=4, ch_int=64, ch_out=64)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)
        # self.se=SEBlock(channel=512)
        self.sa4=SqueezeAttentionBlock(ch_in=512, ch_out=512)
        self.sa3 = SqueezeAttentionBlock(ch_in=256, ch_out=256)
        self.sa2 = SqueezeAttentionBlock(ch_in=128, ch_out=128)
        self.sa1 = SqueezeAttentionBlock(ch_in=64, ch_out=64)

        # deep supervision


        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=patch_size if i == 0 else 2,
                       in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                       embed_dim=self.embed_dims[i],
                       overlapped=overlapped_patch)
            for i in range(self.num_stages)])

        main_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))

            block = nn.ModuleList([
                MySequential(*[
                    ChannelBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        cpe_act=cpe_act
                    ) if attention_type == 'channel' else
                    SpatialBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        cpe_act=cpe_act,
                        window_size=window_size,
                    ) if attention_type == 'spatial' else None
                    for attention_id, attention_type in enumerate(attention_types)]
                             ) for layer_id, item in enumerate(block_param)
            ])
            main_blocks.append(block)
        self.main_blocks = nn.ModuleList(main_blocks)

        # add a norm layer for each output
        for i_layer in range(self.num_stages):
            layer = norm_layer(self.embed_dims[i_layer])  # if i_layer != 0 else nn.Identity()
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

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

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     logger = get_root_logger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        c1=x
        x, size = self.patch_embeds[0](x, (x.size(2), x.size(3)))
        features = [x]
        sizes = [size]
        branches = [0]

        for block_index, block_param in enumerate(self.architecture):
            branch_ids = sorted(set(block_param))
            for branch_id in branch_ids:
                if branch_id not in branches:
                    x, size = self.patch_embeds[branch_id](features[-1], sizes[-1])
                    features.append(x)
                    sizes.append(size)
                    branches.append(branch_id)
            for layer_index, branch_id in enumerate(block_param):
                features[branch_id], _ = self.main_blocks[block_index][layer_index](features[branch_id],
                                                                                    sizes[branch_id])

        outs = []
        for i in range(self.num_stages):
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(features[i])
            H, W = sizes[i]
            out = x_out.view(-1, H, W, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
        x_t=outs[3]
        x_t_1=outs[2]
        x_t_2=outs[1]  #192
        x_t_3=outs[0]  #96
        x_v_t=x_t_3
        x=c1
        # x = self.firstconv(x)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        #
        # x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        # x_u_3 = self.encoder1(x)  # [2, 64, 56, 80]
        # x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        # x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        # x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]
        # x_v_c=x_u_3
        #
        # x_c = self.fusion_c(x_u, x_t)  # 512,1/32
        #
        # #x_c= self.sa4(x_c)
        # x_c_1 = self.fusion_c1(x_u_1, x_t_1)  # 256
        # # x_c_1= self.sa3(x_c_1)
        # x_c_2 = self.fusion_c2(x_u_2, x_t_2)  # 128
        # # x_c_2 = self.sa2(x_c_2)
        # x_c_3 = self.fusion_c3(x_u_3, x_t_3)  # 64
        # # x_c_3 = self.sa1(x_c_3)
        # x_v_f = x_c_3

        d4 = self.decoder4(x_t) + x_t_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d3 = self.decoder3(d4) + x_t_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d2 = self.decoder2(d3) + x_t_3  # [2, 64, 56, 80]
        d1 = self.decoder1(d2)  # [2, 64, 112, 160]
        # d4 = self.decoder4(x_u) + x_u_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        # d3 = self.decoder3(d4) + x_u_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        # d2 = self.decoder2(d3) + x_u_3  # [2, 64, 56, 80]
        # d1 = self.decoder1(d2)  # [2, 64, 112, 160]

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        xv=out
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]
        x_v_d0 = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)     #256
        x_v_d1= x_v_d0
        x_v_d2=x_v_d0


        x_v_d3 = x_v_d0   #128
        x_v_d4 = x_v_d0
        x_v_d5 = x_v_d0

        x_v_d6 = x_v_d0    #64
        x_v_d7 = x_v_d0
        x_v_d8 =x_v_d0

        return out,x_v_d0,x_v_d1,x_v_d2,x_v_d3,x_v_d4,x_v_d5,x_v_d6,x_v_d7,x_v_d8


# from torchstat import stat
# import torchvision.models as models
#
# model = DaViT()
# stat(model, (3, 224, 224))
