import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
# from ..builder import BACKBONES
from torchvision import models
from functools import partial
from torchvision.models import resnet34 as resnet
import numpy as np
from ptflops import get_model_complexity_info
nonlinearity = partial(F.relu, inplace=True)
from lib import pvt_v2
from timm.models.vision_transformer import _cfg
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
class spAttention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int,r_2):
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

        self.fc1 = nn.Conv2d(F_l, F_l // r_2, kernel_size=1)      #//除完取整
        self.silu = nn.SiLU(True)
        self.fc2 = nn.Conv2d(F_l // r_2, F_int, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        #self.spatial = nn.Conv2d(2, 1, kernel_size=7)

    def forward(self, g, x):
        psi = self.relu(g + x)
        psi = self.psi(psi)
        return x * psi
        # f=g+x
        # psi = self.relu(f)
        # p1=psi
        # psi = self.psi(psi)
        # p2=psi
        # p3=x * psi
        # return x * psi,p2,f

class spAttention_block11(nn.Module):
    def __init__(self, F_g, F_l, F_int,r_2):
        super(spAttention_block11, self).__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        f=g+x
        psi = self.relu(f)
        psi = self.psi(psi)
        return x * psi

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
# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super(DecoderBlock, self).__init__()
#         #self.conv1 = ODConv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         #ODConv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nonlinearity
#
#         self.conatt = nn.Conv2d(in_channels, in_channels // 4, 1)
#         # ODConv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.normatt = nn.BatchNorm2d(in_channels // 4)
#         self.reluatt = nonlinearity
#
#         self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
#         self.norm2 = nn.BatchNorm2d(in_channels // 4)
#         self.relu2 = nonlinearity
#
#         self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
#         #self.conv3 = ODConv2d(in_channels // 4, n_filters,  kernel_size=3, padding=1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nonlinearity
#
#         self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv = conv_block(in_channels, in_channels // 4)
#         self.conv_atten = conv_block(in_channels, in_channels // 4)
#         self.upsample = nn.Upsample(scale_factor=2)
#         #self.upsample=F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)
#
#     def forward(self, x):
#         x_res = self.conv1(x)
#         x_res= self.norm1(x_res)
#         x_res = self.relu1(x_res)
#         # x_res = self.conv(x)
#         B,C,H,W=x_res.shape
#         y = self.avg_pool(x)
#         x2 = y
#         y = self.conatt(y)
#         y = self.normatt(y)
#         y = self.reluatt(y)
#         x3 = y
#         y = self.upsample(y)
#         y=F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
#         x = (y * x_res) + y
#         x = self.deconv2(x)
#         x = self.norm2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x
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
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
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
# class BiFusion_block(nn.Module):   #BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
#     def  __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
#                  drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
#         super(BiFusion_block, self).__init__()
#         # coor(ch2,r2)(384,4)trans
#         # self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (256到256）
#         # self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (384，256)
#         self.W_g = Conv(512, 512, 1, bn=True, relu=False)  # (256到256）
#         self.W_x = Conv(512, 512, 1, bn=True, relu=False)  # (384，256)
#         #
#         # self.W_g=odconv1x1(ch_1, ch_int, reduction=0.0625, kernel_num=1)
#         # self.W_x = odconv1x1(ch_2, ch_int, reduction=0.0625, kernel_num=1)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
#         self.dropout = nn.Dropout2d(drop_rate)
#         self.drop_rate = drop_rate
#         self.inplanes = ch_1
#         self.Conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1)
#         r = 8
#         L = 48
#         d = max(int(self.inplanes / r), L)
#         self.W1 = nn.Linear(2 * self.inplanes, d)
#         self.W2 = nn.Linear(d, self.inplanes)
#         self.W3 = nn.Linear(d, self.inplanes)
#
#     def forward(self, g, x):
#
#         # U1 = self.W_g(g)
#         # U2 = self.W_x(x)
#
#         U1 = g
#         U2 = x
#
#         #fuse = g + x
#         U=U1 + U2
#
#
#
#         avg_pool = F.avg_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
#         max_pool = F.max_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
#         # p = torch.mean(torch.mean(U, dim=2), dim=2).reshape(-1, self.inplanes, 1, 1)
#         # q = torch.matmul(
#         #     U.reshape([-1, self.inplanes, x.shape[2] * x.shape[3]]),
#         #     (nn.Softmax(dim=1)(self.Conv_q(x))).reshape([-1, x.shape[2] * x.shape[3], 1])
#         # ).reshape([-1, self.inplanes, 1, 1])
#         sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
#         # s = torch.sigmoid(torch.cat([p, q], 1)).reshape([-1, self.inplanes * 2])
#         # z1 = self.W2(nn.ReLU()(self.W1(s)))
#         # z2 = self.W3(nn.ReLU()(self.W1(s)))
#         zc1 = self.W2(nn.ReLU()(self.W1(sc)))
#         zc2 = self.W3(nn.ReLU()(self.W1(sc)))
#         # a1 = (torch.exp(z1) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
#         # a2 = (torch.exp(z2) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
#         ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
#         ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
#         fuse = U1 * ac1 + U2 * ac2+U
#
#         #fuse = self.residual(torch.cat([g, x], 1))  # x是 trans分支，g是cnn分支，，bp是中间的
#         return fuse

class BiFusion_block(nn.Module):   #BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
    def  __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()
        # self.W_g = Conv(ch_2, ch_1, 1, bn=True, relu=True)  # (256到256）
        # self.W_x = Conv(ch_1, ch_1, 1, bn=True, relu=True)  # (384，256)
        self.W_g = Conv(ch_2, ch_1, 1, bn=False, relu=False)  # (256到256）
        self.W_x = Conv(ch_1, ch_1, 1, bn=False, relu=False)  # (384，256)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = ch_1
        r = 8
        L = 48
        d = max(int(self.inplanes / r), L)
        self.W1 = nn.Linear(2 * self.inplanes, d)
        self.W2 = nn.Linear(d, self.inplanes)
        self.W3 = nn.Linear(d, self.inplanes)

    def forward(self, x, g):   #x,trans
        U1 =self.W_g(g)
        U2 =self.W_x(x)
        U=U1 + U2
        # avg_pool = F.avg_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        # max_pool = F.max_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))

        avg_pool = F.avg_pool2d(U1, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        max_pool = F.max_pool2d(U2, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))

        sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])

        zc1 = self.W2(nn.ReLU()(self.W1(sc)))
        zc2 = self.W3(nn.ReLU()(self.W1(sc)))

        ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        fuse = U1 * ac1 + U2 * ac2

        #fuse = self.residual(torch.cat([g, x], 1))  # x是 trans分支，g是cnn分支，，bp是中间的
        return fuse
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop_out=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.dropout = drop_out

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = nn.Dropout2d(0.5)(x)
        return x
class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 dilation=[1, 3, 5], groups=1, bias=True,
                 act_layer='nn.SiLU(True)', init='kaiming'):
        super().__init__()
        assert in_planes % groups == 0
        assert kernel_size == 3, 'support kernel size 3 now'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.with_bias = bias

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        self.act = eval(act_layer)
        self.init = init
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
        if self.with_bias:
            if self.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.init == 'kaiming':
                bound = self.groups / (self.kernel_size ** 2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = 0
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output


class CSA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, padding=1, stride=2,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim ** -0.5

        self.attn = nn.Linear(in_dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.csa_group = 1
        assert out_dim % self.csa_group == 0
        self.weight = nn.Conv2d(
            self.kernel_size * self.kernel_size * out_dim,
            self.kernel_size * self.kernel_size * out_dim,
            1,
            stride=1, padding=0, dilation=1,
            groups=self.kernel_size * self.kernel_size * self.csa_group,
            bias=qkv_bias,
        )
        assert qkv_bias == False
        fan_out = self.kernel_size * self.kernel_size * self.out_dim
        fan_out //= self.csa_group
        self.weight.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # init

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, v=None):
        B, H, W, _ = x.shape
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = x.permute(0, 3, 1, 2)  # B,C,H, W
        v = self.unfold(v).reshape(
            B, self.out_dim, self.kernel_size * self.kernel_size, h * w
        ).permute(0, 3, 2, 1).reshape(B * h * w, self.kernel_size * self.kernel_size * self.out_dim, 1, 1)
        v = self.weight(v)
        v = v.reshape(B, h * w, self.kernel_size * self.kernel_size, self.num_heads,
                      self.out_dim // self.num_heads).permute(0, 3, 1, 2, 4).contiguous()  # B,H,N,kxk,C/H

        x = (attn @ v).permute(0, 1, 4, 3, 2)
        x = x.reshape(B, self.out_dim * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0., with_depconv=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_depconv = with_depconv

        if self.with_depconv:
            self.fc1 = nn.Conv2d(
                in_features, hidden_features, 1, stride=1, padding=0, dilation=1,
                groups=1, bias=True,
            )
            self.depconv = nn.Conv2d(
                hidden_features, hidden_features, 3, stride=1, padding=1, dilation=1,
                groups=hidden_features, bias=True,
            )
            self.act = act_layer()
            self.fc2 = nn.Conv2d(
                hidden_features, out_features, 1, stride=1, padding=0, dilation=1,
                groups=1, bias=True,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.with_depconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.fc1(x)
            x = self.depconv(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


class Attention(nn.Module):
    def __init__(
            self,
            dim, num_heads=8, qkv_bias=False,
            qk_scale=None, attn_drop=0.,
            proj_drop=0.,
            rasa_cfg=None, sr_ratio=1,
            linear=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.rasa_cfg = rasa_cfg
        self.use_rasa = rasa_cfg is not None
        self.sr_ratio = sr_ratio


        self.atrous_rates = [1, 3, 5]
        self.act_layer = 'nn.SiLU(True)'
        self.r_num=2
        self.init='kaiming'


        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        if self.use_rasa:

            if self.atrous_rates is not None:

                self.ds = ds_conv2d(
                    dim, dim, kernel_size=3, stride=1,
                    dilation=self.atrous_rates, groups=dim, bias=qkv_bias,
                    act_layer=self.act_layer, init=self.init,
                )
            if self.r_num > 1:
                self.silu = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _inner_attention(self, x):
        B, H, W, C = x.shape
        q = self.q(x).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.use_rasa:
            if self.atrous_rates is not None:
                q = q.permute(0, 1, 3, 2).reshape(B, self.dim, H, W).contiguous()
                q = self.ds(q)
                q = q.reshape(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 3, 2).contiguous()

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 3, 1, 2)
                x_ = self.sr(x_).permute(0, 2, 3, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            raise NotImplementedError

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        if self.use_rasa:
            x_in = x
            x = self._inner_attention(x)  #self-attention
            if self.r_num > 1:
                x = self.silu(x)
            for _ in range(self.r_num - 1):
                x = x + x_in
                x_in = x
                x = self._inner_attention(x)
                x = self.silu(x)
        else:
            x = self._inner_attention(x)
        return x


class Transformer_block(nn.Module):
    def __init__(self, dim,
                 num_heads=1, mlp_ratio=3., attn_drop=0.,
                 drop_path=0., sa_layer='sa', rasa_cfg=None, sr_ratio=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None, with_depconv=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sa_layer == 'csa':
            self.attn = CSA(
                dim, dim, num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop)
        elif sa_layer in ['rasa', 'sa']:
            self.attn = Attention(
                dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, rasa_cfg=rasa_cfg, sr_ratio=sr_ratio)
        else:
            raise NotImplementedError
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            with_depconv=with_depconv)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.patch_size[0] == 7:
            x = self.proj(x)   #7x7卷积，c=64，h/4，w/4
            x = x.permute(0, 2, 3, 1)  #b hw  c
            x = self.norm(x)
        else:
            x = x.permute(0, 3, 1, 2)
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        return x

class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''
    def __init__(self, in_ch, s=4):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1,self.s):
            if i == 1:
                channels=in_ch
                acc_channels=channels//2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels=in_ch+acc_channels
                acc_channels=channels//2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]
class ResEncoder_hs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder_hs, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = HSBlock(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out =out+ residual
        out = self.relu(out)
        return out


class RFB_hs(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB_hs, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                         relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                         dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                         dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,strides=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,strides,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool

def downsample_soft():
    return SoftPooling2D(2, 2)
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# 就是rfbnet论文中的fig 4(a)

class BasicConv_hs(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv_hs, self).__init__()
        self.out_channels = out_planes
        self.conv = HSBlock_rfb(in_planes, s=4, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class HSBlock_rfb(nn.Module):
    '''
    替代3x3卷积
    '''
    def __init__(self, in_ch, s=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock_rfb, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1,self.s):
            if i == 1:
                channels=in_ch
                acc_channels=channels//2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels=in_ch+acc_channels
                acc_channels=channels//2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]
class Ms_red_v1(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(Ms_red_v1, self).__init__()
        self.out_size = (224, 320)
        self.enc_input = ResEncoder_hs(channels, 32)
        self.encoder1 = RFB_hs(32, 64)
        self.encoder2 = RFB_hs(64, 128)
        self.encoder3 = RFB_hs(128, 256)
        self.encoder4 = RFB_hs(256, 512)
        self.downsample = downsample_soft()

    # initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)  # [16, 3, 224, 320]-->[16, 32, 224, 320]
        down1 = self.downsample(enc_input)  # [16, 32, 112, 160]    #1/2

        enc1 = self.encoder1(down1)  # [16, 64, 112, 160]
        down2 = self.downsample(enc1)  # [16, 64, 56, 80]           #1/4

        enc2 = self.encoder2(down2)  # [16, 128, 56, 80]
        down3 = self.downsample(enc2)  # [16, 128, 28, 40]          #1/8

        enc3 = self.encoder3(down3)  # [16, 256, 28, 40]
        down4 = self.downsample(enc3)  # [16, 256, 14, 20]          #1/16

        input_feature = self.encoder4(down4)  # [16, 512, 14, 18]   #1/32

        return input_feature

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

class lite(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
                 embed_dims=[64, 64, 160, 256], num_heads=[2, 4, 8, 16],
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=True,
                 mlp_ratios=[4, 8, 4, 4], mlp_depconv=[False, True, True, True], sr_ratios=[0, 4, 2, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True, init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head  # set false for downstream tasks
        self.init_cfg = init_cfg

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

        self.attn_block1 = spAttention_block(256, 256, 256,4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)


        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=160, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=64, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

        network = []
        for stage_idx in range(len(layers)):
            _patch_embed = OverlapPatchEmbed(
                patch_size=7 if stage_idx == 0 else 3,
                stride=4 if stage_idx == 0 else 2,
                in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
            )

            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,  # I am here
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)

            network.append(nn.Sequential(
                _patch_embed,
                _blocks
            ))

        # backbone
        self.backbone = nn.ModuleList(network)

        # classification head
        if self.with_cls_head:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            #             self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx])
            #                                                    for idx in range(len(embed_dims))])
            self.downstream_norms = nn.ModuleList([nn.Identity()
                                                   for idx in range(len(embed_dims))])
        self.apply(self._init_weights)
        self.init_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone(self):
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg
            pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(pretrained, strict=True)

    def forward(self, x):
        c1=x
        outs = []
        for idx, stage in enumerate(self.backbone):  #
            x = stage(x)
            outs.append(x.permute(0, 3, 1, 2).contiguous())

        x = self.norm(x)
        x=x.permute(0, 3, 1, 2).contiguous()
        outs[3] = x

        c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        c1 = self.firstbn(c1)
        c1 = self.firstrelu(c1)
        c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(c1)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]

        x_c = self.fusion_c(x_u, outs[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x_u_1, outs[2])  # 256
        x_c_2 = self.fusion_c2(x_u_2, outs[1])  # 128
        x_c_3 = self.fusion_c3(x_u_3, outs[0])  # 64

        #d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d40 = self.decoder4(x_c)
        d41,p1,ff1=self.attn_block1(d40,x_c_1)
        d4=d40+d41

        # d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d30 = self.decoder3(d4)
        d31,p2,ff2 = self.attn_block2(d30, x_c_2)
        d3 = d30 + d31

        #d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
        d20 = self.decoder2(d3)
        d21,p3,ff3 = self.attn_block3(d20, x_c_3)
        d2 = d20 + d21

        d1 = self.decoder1(d2)  # [2, 64, 112, 160]



        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]


        return out
class lite_vision(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
                 embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16],
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=True,
                 mlp_ratios=[4, 8, 4, 4], mlp_depconv=[False, True, True, True], sr_ratios=[8, 4, 2, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True, init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head  # set false for downstream tasks
        self.init_cfg = init_cfg

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

        self.attn_block1 = spAttention_block(256, 256, 256,4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)


        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64)
        #
        # self.Up5 = up_conv(ch_in=512, ch_out=256)
        # self.Up_conv5 = conv_block(ch_in=512, ch_out=256)
        # self.Up4 = up_conv(ch_in=256, ch_out=128)
        # self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        # self.Up3 = up_conv(ch_in=128, ch_out=64)
        # self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        # self.Up2=nn.Upsample(scale_factor=4)
        # self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

        network = []
        for stage_idx in range(len(layers)):
            _patch_embed = OverlapPatchEmbed(
                patch_size=7 if stage_idx == 0 else 3,
                stride=4 if stage_idx == 0 else 2,
                in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
            )

            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,  # I am here
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)

            network.append(nn.Sequential(
                _patch_embed,
                _blocks
            ))

        # backbone
        self.backbone = nn.ModuleList(network)

        # classification head
        if self.with_cls_head:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            #             self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx])
            #                                                    for idx in range(len(embed_dims))])
            self.downstream_norms = nn.ModuleList([nn.Identity()
                                                   for idx in range(len(embed_dims))])
        self.apply(self._init_weights)
        self.init_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone(self):
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg
            pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(pretrained, strict=True)
    # def init_backbone(self):
    #
    #     assert 'checkpoint' in self.init_cfg
    #     pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
    #     self.load_state_dict(pretrained, strict=True)


    # def forward(self, x):
    #     c1=x
    #     d1=x
    #     outs = []
    #     c=x
    #     for idx, stage in enumerate(self.backbone):  #
    #         x = stage(x)
    #         outs.append(x.permute(0, 3, 1, 2).contiguous())
    #
    #     x = self.norm(x)
    #     x=x.permute(0, 3, 1, 2).contiguous()
    #     outs[3] = x
    #     c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
    #     c1 = self.firstbn(c1)
    #     c1 = self.firstrelu(c1)
    #     c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
    #     x_u_3 = self.encoder1(c1)  # [2, 64, 56, 80]
    #     x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
    #     x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
    #     x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]
    #
    #     x_c = self.fusion_c(x_u, outs[3])  # 512,1/32
    #     x_c_1 = self.fusion_c1(x_u_1, outs[2])  # 256
    #     x_c_2 = self.fusion_c2(x_u_2, outs[1])  # 128
    #     x_c_3 = self.fusion_c3(x_u_3, outs[0])  # 64
    #
    #     d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
    #     d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
    #     d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
    #     d1 = self.decoder1(d2)  # [2, 64, 112, 160]
    #
    #     out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
    #     out = self.finalrelu1(out)
    #     out = self.finalconv2(out)  # [2, 32, 224, 320]
    #     out = self.finalrelu2(out)
    #     out = self.finalconv3(out)  # [2, 2, 224, 320]
    #
    #
    #
    #     # x_v_d2 = F.interpolate(self.vconv2(x_u), size=(224, 224), mode='bilinear', align_corners=False)
    #     # x_v_d3 = F.interpolate(self.vconv3(outs[3]), size=(224, 224), mode='bilinear', align_corners=False)
    #     #
    #     #
    #     # x_v_d4 = F.interpolate(self.vconv4(x_u_1), size=(224, 224), mode='bilinear', align_corners=False)
    #     # x_v_d5= F.interpolate(self.vconv5(outs[2]), size=(224, 224), mode='bilinear', align_corners=False)
    #
    #
    #     # x_v_d6 = F.interpolate(self.vconv6(x_u_2), size=(224, 224), mode='bilinear', align_corners=False)
    #     # x_v_d7 = F.interpolate(self.vconv7(outs[1]), size=(224, 224), mode='bilinear', align_corners=False)
    #     # x_v_d8 = F.interpolate(self.vconv8(x_c_2), size=(224, 224), mode='bilinear', align_corners=False)
    #
    #
    #
    #     x_v_d0 = F.interpolate(x_u_1, size=(224, 224), mode='bilinear', align_corners=False)
    #     x_v_d1= F.interpolate(outs[2], size=(224, 224), mode='bilinear', align_corners=False)
    #     x_v_d2=F.interpolate(x_c_1, size=(224, 224), mode='bilinear', align_corners=False)
    #
    #
    #     x_v_d3 = F.interpolate(x_u_2, size=(224, 224), mode='bilinear', align_corners=False)
    #     x_v_d4 = F.interpolate(outs[1], size=(224, 224), mode='bilinear', align_corners=False)
    #     x_v_d5 = F.interpolate(x_c_2, size=(224, 224), mode='bilinear', align_corners=False)
    #
    #     x_v_d6 = F.interpolate(x_u_3, size=(224, 224), mode='bilinear', align_corners=False)
    #     x_v_d7 = F.interpolate(outs[0], size=(224, 224), mode='bilinear', align_corners=False)
    #     x_v_d8 = F.interpolate(x_c_3, size=(224, 224), mode='bilinear', align_corners=False)
    #
    #
    #
    #     return out,x_v_d0,x_v_d1,x_v_d2,x_v_d3,x_v_d4,x_v_d5,x_v_d6,x_v_d7,x_v_d8
    def forward(self, x):
        c1=x
        outs = []
        for idx, stage in enumerate(self.backbone):  #
            x = stage(x)
            outs.append(x.permute(0, 3, 1, 2).contiguous())

        x = self.norm(x)
        x=x.permute(0, 3, 1, 2).contiguous()
        outs[3] = x

        c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        c1 = self.firstbn(c1)
        c1 = self.firstrelu(c1)
        c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(c1)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]

        x_c = self.fusion_c(x_u, outs[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x_u_1, outs[2])  # 256
        x_c_2 = self.fusion_c2(x_u_2, outs[1])  # 128
        x_c_3 = self.fusion_c3(x_u_3, outs[0])  # 64

        #d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d40 = self.decoder4(x_c)
        d41,p1,ff1=self.attn_block1(d40,x_c_1)
        d4=d40+d41

        # d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d30 = self.decoder3(d4)
        d31,p2,ff2 = self.attn_block2(d30, x_c_2)
        d3 = d30 + d31

        #d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
        d20 = self.decoder2(d3)
        d21,p3,ff3 = self.attn_block3(d20, x_c_3)
        d2 = d20 + d21

        d1 = self.decoder1(d2)  # [2, 64, 112, 160]


        #decoding + concat path
        # d5 = self.Up5(x_c)
        # d5 = torch.cat((x_c_1, d5), dim=1)
        # d5 = self.Up_conv5(d5)
        #
        # d4 = self.Up4(d5)
        # d4 = torch.cat((x_c_2, d4), dim=1)
        # d4 = self.Up_conv4(d4)
        #
        # d3 = self.Up3(d4)
        # d3 = torch.cat((x_c_3, d3), dim=1)
        # d3 = self.Up_conv3(d3)
        #
        # d3=self.Up2(d3)
        #
        # out = self.Conv_1x1(d3)


        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        # x_v_d2 = F.interpolate(self.vconv2(x_u), size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d3 = F.interpolate(self.vconv3(outs[3]), size=(224, 224), mode='bilinear', align_corners=False)
        #
        #
        # x_v_d4 = F.interpolate(self.vconv4(x_u_1), size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d5= F.interpolate(self.vconv5(outs[2]), size=(224, 224), mode='bilinear', align_corners=False)


        # x_v_d6 = F.interpolate(self.vconv6(x_u_2), size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d7 = F.interpolate(self.vconv7(outs[1]), size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d8 = F.interpolate(self.vconv8(x_c_2), size=(224, 224), mode='bilinear', align_corners=False)



        # x_v_d0 = F.interpolate(x_u, size=(224, 224), mode='bilinear', align_corners=False)     #256
        # x_v_d1= F.interpolate(outs[3], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d2=F.interpolate(x_c, size=(224, 224), mode='bilinear', align_corners=False)
        #
        # xnmew = x_u_3 + outs[0]
        # x_v_d3 = F.interpolate(x_u_2, size=(224, 224), mode='bilinear', align_corners=False)    #128
        # x_v_d4 = F.interpolate(outs[1], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d5 = F.interpolate(xnmew, size=(224, 224), mode='bilinear', align_corners=False)
        #
        #
        #
        # x_v_d6 = F.interpolate(x_u_3, size=(224, 224), mode='bilinear', align_corners=False)    #64
        # x_v_d7 = F.interpolate(outs[0], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d8 = F.interpolate(x_c_3, size=(224, 224), mode='bilinear', align_corners=False)


        # x_v_d0 = F.interpolate(down5, size=(224, 224), mode='bilinear', align_corners=False)     #256
        # x_v_d1= F.interpolate(outs[3], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d2=F.interpolate(x_c, size=(224, 224), mode='bilinear', align_corners=False)
        #
        #
        # x_v_d3 = F.interpolate(down4, size=(224, 224), mode='bilinear', align_corners=False)    #128
        # x_v_d4 = F.interpolate(outs[1], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d5 = F.interpolate(x_c_2, size=(224, 224), mode='bilinear', align_corners=False)
        #
        # x_v_d6 = F.interpolate(down3, size=(224, 224), mode='bilinear', align_corners=False)    #64
        # x_v_d7 = F.interpolate(outs[0], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d8 = F.interpolate(x_c_3, size=(224, 224), mode='bilinear', align_corners=False)

        # #decoder
        # x_v_d0 = F.interpolate(p1, size=(224, 224), mode='bilinear', align_corners=False)     #256
        # x_v_d1= F.interpolate(ff1, size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d2=F.interpolate(d4, size=(224, 224), mode='bilinear', align_corners=False)
        #
        #
        # x_v_d3 = F.interpolate(p2, size=(224, 224), mode='bilinear', align_corners=False)    #128
        # x_v_d4 = F.interpolate(ff2, size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d5 = F.interpolate(d3, size=(224, 224), mode='bilinear', align_corners=False)
        #
        # x_v_d6 = F.interpolate(p3, size=(224, 224), mode='bilinear', align_corners=False)    #64
        # x_v_d7 = F.interpolate(ff3, size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d8 = F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)

        # decoder
        # x_v_d0 = F.interpolate(p1, size=(64, 64), mode='bilinear', align_corners=False)  # 256
        # x_v_d1 = F.interpolate(p1, size=(64, 64), mode='bilinear', align_corners=False)
        # x_v_d2 = F.interpolate(p1, size=(64, 64), mode='bilinear', align_corners=False)
        #
        # x_v_d3 = F.interpolate(p2, size=(64, 64), mode='bilinear', align_corners=False)  # 128
        # x_v_d4 = F.interpolate(p2, size=(64, 64), mode='bilinear', align_corners=False)
        # x_v_d5 = F.interpolate(p2, size=(64, 64), mode='bilinear', align_corners=False)
        #
        # x_v_d6 = F.interpolate(p3, size=(64, 64), mode='bilinear', align_corners=False)  # 64
        # x_v_d7 = F.interpolate(p3, size=(64, 64), mode='bilinear', align_corners=False)
        # x_v_d8 = F.interpolate(p3, size=(64, 64), mode='bilinear', align_corners=False)

        #fusion
        # x_v_d0 = F.interpolate(x_u_1, size=(224, 224), mode='bilinear', align_corners=False)     #256
        # x_v_d1= F.interpolate(outs[2], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d2=F.interpolate(x_c_1, size=(224, 224), mode='bilinear', align_corners=False)
        #
        #
        # x_v_d3 = F.interpolate(x_u_2, size=(224, 224), mode='bilinear', align_corners=False)    #128
        # x_v_d4 = F.interpolate(outs[2], size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d5 = F.interpolate(x_c_2, size=(224, 224), mode='bilinear', align_corners=False)
        #
        # x_v_d6 = F.interpolate(p3, size=(224, 224), mode='bilinear', align_corners=False)    #64
        # x_v_d7 = F.interpolate(ff3, size=(224, 224), mode='bilinear', align_corners=False)
        # x_v_d8 = F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)



        # return out,x_v_d0,x_v_d1,x_v_d2,x_v_d3,x_v_d4,x_v_d5,x_v_d6,x_v_d7,x_v_d8
        return out
class Doublelite_vision(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
                 embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16],
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=True,
                 mlp_ratios=[4, 8, 4, 4], mlp_depconv=[False, True, True, True], sr_ratios=[8, 4, 2, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True, init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head  # set false for downstream tasks
        self.init_cfg = init_cfg

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1

        self.firstbn = resnet.bn1

        self.firstrelu = resnet.relu

        self.firstmaxpool = resnet.maxpool
        # downsampling
        self.conv = conv_block(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = conv_block(64, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(64, 160)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(160, 256, drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.attn_block1 = spAttention_block(256, 256, 256,4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)



        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2=nn.Upsample(scale_factor=4)

        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)



        network = []
        for stage_idx in range(len(layers)):
            _patch_embed = OverlapPatchEmbed(
                patch_size=7 if stage_idx == 0 else 3,
                stride=4 if stage_idx == 0 else 2,
                in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
            )

            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,  # I am here
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)

            network.append(nn.Sequential(
                _patch_embed,
                _blocks
            ))

        # backbone
        self.backbone = nn.ModuleList(network)

        # classification head
        if self.with_cls_head:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            #             self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx])
            #                                                    for idx in range(len(embed_dims))])
            self.downstream_norms = nn.ModuleList([nn.Identity()
                                                   for idx in range(len(embed_dims))])
        self.apply(self._init_weights)
        self.init_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone(self):
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg
            pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(pretrained, strict=True)

    def forward(self, x):
        c1=x
        img=x
        outs = []
        # for idx, stage in enumerate(self.backbone):  #
        #     x = stage(x)
        #     outs.append(x.permute(0, 3, 1, 2).contiguous())
        for idx, stage in enumerate(self.backbone):  #
            x = stage(x)
            if idx==0:
                xf = x.permute(0, 3, 1, 2).contiguous()
                c1 = self.firstconv(img)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
                c1 = self.firstbn(c1)
                c1 = self.firstrelu(c1)
                c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
                c1 = self.encoder1(c1)# [2, 64, 56, 80]
               # [2, 64, 56, 80]   1/4
                fuse1 = self.fusion_c3(c1, xf)  # 64
                x=fuse1.permute(0, 2, 3, 1).contiguous()
            elif idx == 1:
                xf1 = x.permute(0, 3, 1, 2).contiguous()
                c1 = self.encoder2(fuse1)
                # [2, 128, 28, 40]   1/8
                fuse2 = self.fusion_c2(c1, xf1)
                x = fuse2.permute(0, 2, 3, 1).contiguous()
            elif idx == 2:
                xf2 = x.permute(0, 3, 1, 2).contiguous()
                c1 = self.encoder3(fuse2)  # [2, 256, 28, 40]  1/16
                fuse3 = self.fusion_c1(c1, xf2)
                x = fuse3.permute(0, 2, 3, 1).contiguous()
            elif idx == 3:
                x = self.norm(x)
                xf3 = x.permute(0, 3, 1, 2).contiguous()
                c1 = self.encoder4(fuse3)
                fuse4 = self.fusion_c(c1, xf3)

        d40 = self.decoder4(fuse4)
        d41,p1,ff1=self.attn_block1(d40,fuse3)
        d4=d40+d41

        d30 = self.decoder3(d4)
        d31,p2,ff2 = self.attn_block2(d30, fuse2)
        d3 = d30 + d31

        #d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
        d20 = self.decoder2(d3)
        d21,p3,ff3 = self.attn_block3(d20, fuse1)
        d2 = d20 + d21

        d1 = self.decoder1(d2)  # [2, 64, 112, 160]

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        return out
# class lite_vision(nn.Module):
#
#     def __init__(self, layers=[2, 2, 2, 2], in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
#                  embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16],
#                  sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=True,
#                  mlp_ratios=[4, 8, 4, 4], mlp_depconv=[False, True, True, True], sr_ratios=[8, 4, 2, 1],
#                  qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True, init_cfg=None):
#
#         super().__init__()
#         self.embed_dims = embed_dims
#         self.num_heads = num_heads
#         self.mlp_depconv = mlp_depconv
#         self.sr_ratios = sr_ratios
#         self.layers = layers
#         self.num_classes = num_classes
#         self.sa_layers = sa_layers
#         self.rasa_cfg = rasa_cfg
#         self.with_cls_head = with_cls_head  # set false for downstream tasks
#         self.init_cfg = init_cfg
#
#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#
#         self.firstconv = resnet.conv1
#
#         self.firstbn = resnet.bn1
#
#         self.firstrelu = resnet.relu
#
#         self.firstmaxpool = resnet.maxpool
#
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         # self.enc_input = ResEncoder_hs(channels, 32)
#         # self.encoder1 = RFB_hs(32, 64)
#         # self.encoder2 = RFB_hs(64, 128)
#         # self.encoder3 = RFB_hs(128, 256)
#         # self.encoder4 = RFB_hs(256, 512)
#         # self.downsample = downsample_soft()
#
#
#         self.attn_block1 = spAttention_block(256, 256, 256,4)
#         self.attn_block2 = spAttention_block(128, 128, 128, 4)
#         self.attn_block3 = spAttention_block(64, 64, 64, 4)
#
#
#
#         self.decoder4 = DecoderBlock(512, filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
#         self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)
#         self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=128, ch_out=128)
#         self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64)
#
#         self.Up5 = up_conv(ch_in=512, ch_out=256)
#         self.Up_conv5 = conv_block(ch_in=512, ch_out=256)
#
#         self.Up4 = up_conv(ch_in=256, ch_out=128)
#         self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
#
#         self.Up3 = up_conv(ch_in=128, ch_out=64)
#         self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
#
#         self.Up2=nn.Upsample(scale_factor=4)
#
#         self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
#
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)
#
#         self.final_x = nn.Sequential(
#             Conv(512, 64, 1, bn=True, relu=True),
#             Conv(64, 64, 3, bn=True, relu=True),
#             Conv(64, 1, 3, bn=False, relu=False)
#             )
#
#         self.final_1 = nn.Sequential(
#             Conv(64, 64, 3, bn=True, relu=True),
#             Conv(64, 1, 3, bn=False, relu=False)
#             )
#
#         self.final_2 = nn.Sequential(
#             Conv(64, 64, 3, bn=True, relu=True),
#             Conv(64, 1, 3, bn=False, relu=False)
#             )
#
#
#
#         network = []
#         for stage_idx in range(len(layers)):
#             _patch_embed = OverlapPatchEmbed(
#                 patch_size=7 if stage_idx == 0 else 3,
#                 stride=4 if stage_idx == 0 else 2,
#                 in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
#                 embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
#             )
#
#             _blocks = []
#             for block_idx in range(layers[stage_idx]):
#                 block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
#                 _blocks.append(Transformer_block(
#                     embed_dims[stage_idx],
#                     num_heads=num_heads[stage_idx],
#                     mlp_ratio=mlp_ratios[stage_idx],
#                     sa_layer=sa_layers[stage_idx],
#                     rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,  # I am here
#                     sr_ratio=sr_ratios[stage_idx],
#                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     attn_drop=attn_drop_rate, drop_path=block_dpr,
#                     with_depconv=mlp_depconv[stage_idx]))
#             _blocks = nn.Sequential(*_blocks)
#
#             network.append(nn.Sequential(
#                 _patch_embed,
#                 _blocks
#             ))
#
#         # backbone
#         self.backbone = nn.ModuleList(network)
#
#         # classification head
#         if self.with_cls_head:
#             self.norm = norm_layer(embed_dims[-1])
#             self.head = nn.Linear(
#                 embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
#         else:
#             #             self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx])
#             #                                                    for idx in range(len(embed_dims))])
#             self.downstream_norms = nn.ModuleList([nn.Identity()
#                                                    for idx in range(len(embed_dims))])
#         self.apply(self._init_weights)
#         self.init_backbone()
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def init_backbone(self):
#         if self.init_cfg is not None:
#             assert 'checkpoint' in self.init_cfg
#             pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
#             self.load_state_dict(pretrained, strict=True)
#
#     def forward(self, x):
#         c1=x
#         d1=x
#         outs = []
#         c=x
#         for idx, stage in enumerate(self.backbone):  #
#             x = stage(x)
#             outs.append(x.permute(0, 3, 1, 2).contiguous())
#
#         x = self.norm(x)
#         x=x.permute(0, 3, 1, 2).contiguous()
#         outs[3] = x
#
#         c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
#         c1 = self.firstbn(c1)
#         c1 = self.firstrelu(c1)
#         c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
#         x_u_3 = self.encoder1(c1)  # [2, 64, 56, 80]
#         x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
#         x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
#         x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]
#
#         x_c = self.fusion_c(x_u, outs[3])  # 512,1/32
#
#         x_c_1 = self.fusion_c1(x_u_1, outs[2])  # 256
#         x_c_2 = self.fusion_c2(x_u_2, outs[1])  # 128
#         x_c_3 = self.fusion_c3(x_u_3, outs[0])  # 64
#
#         #d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
#         d40 = self.decoder4(x_c)
#         d41,p1,ff1=self.attn_block1(d40,x_c_1)
#         d4=d40+d41
#
#         # d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
#         d30 = self.decoder3(d4)
#         d31,p2,ff2 = self.attn_block2(d30, x_c_2)
#         d3 = d30 + d31
#
#         #d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
#         d20 = self.decoder2(d3)
#         d21,p3,ff3 = self.attn_block3(d20, x_c_3)
#         d2 = d20 + d21
#
#         d1 = self.decoder1(d2)  # [2, 64, 112, 160]
#         out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)  # [2, 32, 224, 320]
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)  # [2, 2, 224, 320]
#
#         map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear', align_corners=True)
#         map_1 = F.interpolate(self.final_1(x_c_3), scale_factor=4, mode='bilinear', align_corners=True)
#
#         return map_x, map_1, out

class lite_vision1(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
                 embed_dims=[64, 64, 160, 256], num_heads=[2, 2, 5, 8],
                 sa_layers=['rasa', 'rasa', 'rasa', 'rasa'], rasa_cfg=None,
                 mlp_ratios=[4, 8, 4, 4], mlp_depconv=[False, True, True, True], sr_ratios=[0, 4, 2, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True, init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head  # set false for downstream tasks
        self.init_cfg = init_cfg

        filters = [64, 64, 160, 256]
        # resnet = models.resnet34(pretrained=True)
        #
        # self.firstconv = resnet.conv1
        #
        # self.firstbn = resnet.bn1
        #
        # self.firstrelu = resnet.relu
        #
        # self.firstmaxpool = resnet.maxpool
        #
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4

        # self.attn_block1 = spAttention_block11(256, 256, 256,4)
        # self.attn_block2 = spAttention_block11(128, 128, 128, 4)
        # self.attn_block3 = spAttention_block11(64, 64, 64, 4)



        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
        # self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)
        # self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=128, ch_out=128)
        # self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)



        network = []
        for stage_idx in range(len(layers)):
            _patch_embed = OverlapPatchEmbed(
                patch_size=7 if stage_idx == 0 else 3,
                stride=4 if stage_idx == 0 else 2,
                in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
            )

            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,  # I am here
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)

            network.append(nn.Sequential(
                _patch_embed,
                _blocks
            ))

        # backbone
        self.backbone = nn.ModuleList(network)

        # classification head
        if self.with_cls_head:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            #             self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx])
            #                                                    for idx in range(len(embed_dims))])
            self.downstream_norms = nn.ModuleList([nn.Identity()
                                                   for idx in range(len(embed_dims))])
        self.apply(self._init_weights)
        self.init_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone(self):
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg
            pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(pretrained, strict=True)


    def forward(self, x):


        c1=x
        d1=x
        outs = []
        c=x
        for idx, stage in enumerate(self.backbone):  #
            x = stage(x)
            outs.append(x.permute(0, 3, 1, 2).contiguous())

        x = self.norm(x)
        x=x.permute(0, 3, 1, 2).contiguous()
        outs[3] = x

        # c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        # c1 = self.firstbn(c1)
        # c1 = self.firstrelu(c1)
        # c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
        # x_u_3 = self.encoder1(c1)  # [2, 64, 56, 80]
        # x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        # x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        # x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]

        # x_c = self.fusion_c(x_u, outs[3])  # 512,1/32
        #
        # x_c_1 = self.fusion_c1(x_u_1, outs[2])  # 256
        # x_c_2 = self.fusion_c2(x_u_2, outs[1])  # 128
        # x_c_3 = self.fusion_c3(x_u_3, outs[0])  # 64
        # x_c = self.fusion_c(outs[3], outs[3])  # 512,1/32
        #
        # x_c_1 = self.fusion_c1(outs[2], outs[2])  # 256
        # x_c_2 = self.fusion_c2(outs[1], outs[1])  # 128
        # x_c_3 = self.fusion_c3(outs[0], outs[0])  # 64

        d40 = self.decoder4(outs[3])
        # d41=self.attn_block1(d40,outs[2])
        d4=d40+outs[2]

        d30 = self.decoder3(outs[2])
        # d31= self.attn_block2(d30, outs[1])
        d3 = d30 + outs[1]

        d20 = self.decoder2(outs[1])
        # d21= self.attn_block3(d20, outs[0])
        d2 = d20 + outs[0]

        d1 = self.decoder1(d2)  # [2, 64, 112, 160]

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        return out

class plite_visiondeep(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        # checkpoint = torch.load("D:/PVT-2/PVT-2/pvt_v2_b0.pth")
        checkpoint = torch.load("D:/PVT-2/PVT-2/pre/pvt_v2_b1.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # filters = [32, 64, 160, 256]
        filters = [64, 128, 256, 512]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        fi = [64, 128, 320, 512]

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=fi[3], r_2=4, ch_int=filters[3], ch_out=filters[3])
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=fi[2], r_2=4, ch_int=filters[2], ch_out=filters[2])
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=fi[1], r_2=4, ch_int=filters[1], ch_out=filters[1])
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=fi[0], r_2=4, ch_int=filters[0], ch_out=filters[0])

        self.attn_block1 = spAttention_block(filters[2], filters[2], filters[2], 4)
        self.attn_block2 = spAttention_block(filters[1], filters[1], filters[1], 4)
        self.attn_block3 = spAttention_block(filters[0], filters[0], filters[0], 4)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)
        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        self.Up2=nn.Upsample(scale_factor=4)
        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

        self.final_x = nn.Sequential(
            Conv(512, 256, 1, bn=True, relu=True),
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)

            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        img = x
        pyramid = self.get_pyramid(x)
        x = self.firstconv(img)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        e1 = self.encoder1(x)  # [2, 64, 56, 80]
        e2 = self.encoder2(e1)  # [2, 128, 28, 40]
        e3 = self.encoder3(e2)  # [2, 256, 14, 20]
        e4 = self.encoder4(e3)  # [2, 512, 7, 10]
        # Decoder
        x_c = self.fusion_c(e4, pyramid[3])  # 512,1/32
        x_c_1 = self.fusion_c1(e3, pyramid[2])  # 256
        x_c_2 = self.fusion_c2(e2, pyramid[1])  # 128
        x_c_3 = self.fusion_c3(e1, pyramid[0])  # 64

        # #decoding + concat path
        # d5 = self.Up5(x_c)
        # d5 = torch.cat((x_c_1, d5), dim=1)
        # d5 = self.Up_conv5(d5)
        #
        # d4 = self.Up4(d5)
        # d4 = torch.cat((x_c_2, d4), dim=1)
        # d4 = self.Up_conv4(d4)
        #
        # d3 = self.Up3(d4)
        # d3 = torch.cat((x_c_3, d3), dim=1)
        # d3 = self.Up_conv3(d3)
        # d2=d3
        #
        # d3=self.Up2(d3)
        #
        # out = self.Conv_1x1(d3)

        d40 = self.decoder4(x_c)
        d41 = self.attn_block1(d40, x_c_1)
        d4 = d40 + d41
        d30 = self.decoder3(d4)
        d31 = self.attn_block2(d30, x_c_2)
        d3 = d30 + d31
        d20 = self.decoder2(d3)
        d21 = self.attn_block3(d20, x_c_3)
        d2 = d20 + d21
        d1 = self.decoder1(d2)

        # # Decoder
        # d4 = self.decoder4(e4) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        # d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        # d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]
        # d1 = self.decoder1(d2)       # [2, 64, 112, 160]
        #
        #
        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(d2), scale_factor=4, mode='bilinear', align_corners=True)

        # d2 = F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)

        return map_x,map_1,out
class lite_visiondeep(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
                 embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16],
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=True,
                 mlp_ratios=[4, 8, 4, 4], mlp_depconv=[False, True, True, True], sr_ratios=[8, 4, 2, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True, init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head  # set false for downstream tasks
        self.init_cfg = init_cfg

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

        self.attn_block1 = spAttention_block(256, 256, 256,4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)


        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

        self.final_x = nn.Sequential(
            Conv(512, 256, 1, bn=True, relu=True),
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )


        network = []
        for stage_idx in range(len(layers)):
            _patch_embed = OverlapPatchEmbed(
                patch_size=7 if stage_idx == 0 else 3,
                stride=4 if stage_idx == 0 else 2,
                in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
                embed_dim=embed_dims[0] if stage_idx == 0 else embed_dims[stage_idx],
            )

            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                _blocks.append(Transformer_block(
                    embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    sa_layer=sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg if sa_layers[stage_idx] == 'rasa' else None,  # I am here
                    sr_ratio=sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=mlp_depconv[stage_idx]))
            _blocks = nn.Sequential(*_blocks)

            network.append(nn.Sequential(
                _patch_embed,
                _blocks
            ))

        # backbone
        self.backbone = nn.ModuleList(network)

        # classification head
        if self.with_cls_head:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            #             self.downstream_norms = nn.ModuleList([norm_layer(embed_dims[idx])
            #                                                    for idx in range(len(embed_dims))])
            self.downstream_norms = nn.ModuleList([nn.Identity()
                                                   for idx in range(len(embed_dims))])
        self.apply(self._init_weights)
        self.init_backbone()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone(self):
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg
            pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(pretrained, strict=True)

    def forward(self, x):
        c1=x
        outs = []
        for idx, stage in enumerate(self.backbone):  #
            x = stage(x)
            outs.append(x.permute(0, 3, 1, 2).contiguous())

        x = self.norm(x)
        x=x.permute(0, 3, 1, 2).contiguous()
        outs[3] = x

        c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        c1 = self.firstbn(c1)
        c1 = self.firstrelu(c1)
        c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(c1)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]

        x_c = self.fusion_c(x_u, outs[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x_u_1, outs[2])  # 256
        x_c_2 = self.fusion_c2(x_u_2, outs[1])  # 128
        x_c_3 = self.fusion_c3(x_u_3, outs[0])  # 64


        xv1 = F.interpolate(x_c, size=(224, 224), mode='bilinear', align_corners=False)


        #d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d40 = self.decoder4(x_c)
        d41=self.attn_block1(d40,x_c_1)
        d4=d40+d41

        xv2 = F.interpolate(x_c_1, size=(224, 224), mode='bilinear', align_corners=False)
        xv3 = F.interpolate(d4, size=(224, 224), mode='bilinear', align_corners=False)

        # d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d30 = self.decoder3(d4)
        d31= self.attn_block2(d30, x_c_2)
        d3 = d30 + d31

        xv4 = F.interpolate(x_c_2, size=(224, 224), mode='bilinear', align_corners=False)
        xv5 = F.interpolate(d3, size=(224, 224), mode='bilinear', align_corners=False)

        #d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]    1/4
        d20 = self.decoder2(d3)
        d21= self.attn_block3(d20, x_c_3)
        d2 = d20 + d21

        xv6 = F.interpolate(x_c_1, size=(224, 224), mode='bilinear', align_corners=False)
        xv7 = F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)


        d1 = self.decoder1(d2)  # [2, 64, 112, 160]


        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(d2), scale_factor=4, mode='bilinear', align_corners=True)

        # d2 = F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)


        return map_x,map_1,out,xv1,xv2,xv3,xv4,xv5,xv6,xv7
class baseline(nn.Module):

    def __init__(self, in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True, init_cfg=None,bilinear=True,pretrained=True):

        super().__init__()
        self.with_cls_head = with_cls_head  # set false for downstream tasks
        self.init_cfg = init_cfg
        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-333f7ec4.pth'))
        self.resnet.fc = nn.Identity()
        #self.resnet.layer4 = nn.Identity()



        self.attn_block1 = spAttention_block(256, 256, 256,4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)

        self.up1 = UUp(512, 256, bilinear)
        self.up2 = UUp(256, 128, bilinear)
        self.up3 = UUp(128, 64, bilinear)
        self.up4 = UUp(64, 64, bilinear)
        self.fiup=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(64, 1)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_backbone(self):
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg
            pretrained = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(pretrained, strict=True)

    def forward(self, x):

        # top-down path
        x_u = self.resnet.conv1(x)
        x1=x_u
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_0 = self.resnet.layer3(x_u_1)
        x_u=self.resnet.layer4(x_u_0)

        dx = self.up1(x_u, x_u_0)
        dx = self.up2(dx, x_u_1)
        dx = self.up3(dx, x_u_2)
        dx = self.up4(dx, x1)
        dx=self.fiup(dx)
        out = self.outc(dx)



        x_v_d0 = F.interpolate(dx, size=(224, 224), mode='bilinear', align_corners=False)     #256
        x_v_d1= x_v_d0
        x_v_d2=x_v_d0


        x_v_d3 = x_v_d0   #128
        x_v_d4 = x_v_d0
        x_v_d5 = x_v_d0

        x_v_d6 = x_v_d0    #64
        x_v_d7 = x_v_d0
        x_v_d8 =x_v_d0

        return out,x_v_d0,x_v_d1,x_v_d2,x_v_d3,x_v_d4,x_v_d5,x_v_d6,x_v_d7,x_v_d8

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
        logits = self.outc(x)
        return logits
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

        self.conv = DConv(in_channels+out_channels, out_channels)

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



# @BACKBONES.register_module()
# class lvt(lite_vision_transformer):
#     def __init__(self, rasa_cfg=None, with_cls_head=True, init_cfg=None):
#         super().__init__(
#             layers=[2, 2, 2, 2],
#             patch_size=4,
#             embed_dims=[64, 64, 160, 256],
#             num_heads=[2, 2, 5, 8],
#             mlp_ratios=[4, 8, 4, 4],
#             mlp_depconv=[False, True, True, True],
#             sr_ratios=[8, 4, 2, 1],
#             sa_layers=['csa', 'rasa', 'rasa', 'rasa'],
#             rasa_cfg=rasa_cfg,
#             with_cls_head=with_cls_head,
#             init_cfg=init_cfg,
#         )


from torchstat import stat

import time

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



if __name__ == '__main__':
    # net = lite_vision().cuda()
    # data = torch.randn((1, 3, 224, 224)).cuda()
    # measure_inference_speed(net, (data,))
    # model = lite_vision1()
    # img=torch.randn(1, 3, 224, 224)
    # out=model(img)
    # torch.cuda.synchronize()
    # start = time.time()
    # result = model(img)
    # torch.cuda.synchronize()
    # end = time.time()
    # print('infer_time:', end - start)


    net = plite_visiondeep() #可以为自己搭建的模型
    flops, params = get_model_complexity_info(net, (3,224,224), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)


    #正统计算
    # print('==> Building model..')
    # model = Doublelite_vision()
    # input = torch.randn(1, 3, 224, 224)
    # out=model(input)
    # flops, params = profile(model, (input,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

    # torch.cuda.synchronize()
    # start = time.time()
    # result = model(img.to(device))
    # torch.cuda.synchronize()
    # end = time.time()
    # print('infer_time:', end - start)


    # input = torch.rand(2, 3, 224, 320)
    # # model = RCEM(3, 64)
    # # model = BasicBlock(3, 64, 64)
    # model = lite_vision_transformer()
    # out12 = model(input)
    # print(out12)


