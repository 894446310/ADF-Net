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
#from torchvision.models import resnet34 as resnet

from torchvision.models import resnet50
from torchvision import models


# Attention gate代码
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)

        return x * psi
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

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
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
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu1(x)
        # x_res = self.conv(x)
        # x_res = x
        # y = self.avg_pool(x)
        # x2 = y
        # y = self.conv_atten(y)
        # x3 = y
        # y = self.upsample(y)
        # x = (y * x_res) + y
        # x = self.deconv2(x)
        # x = self.norm2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.norm3(x)
        # x = self.relu3(x)
        return x

def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)



def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

# ODConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,
#                     reduction=0.0625n, kernel_num=1)
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)
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

class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity=x
        x=self.conv(x)
        out=x+identity
        return out

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
class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)
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
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (256到256）
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (384，256)
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

        #fuse = g + x
        U=U1 + U2



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
        fuse = U1 * ac1 + U2 * ac2+U
        # fuse =U



        #fuse = self.residual(torch.cat([g, x], 1))  # x是 trans分支，g是cnn分支，，bp是中间的
        return fuse
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1] #B head N C
                k2, v2 = kv2[0], kv2[1]
                attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

                x = torch.cat([x1,x2], dim=-1)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                        transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
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
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, 64, 7, 2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True)]
        for i in range(num):
            stem.append(nn.Conv2d(64, 64, 3, 1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(64))
            stem.append(nn.ReLU(True))
        stem.append(nn.Conv2d(64, 64, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(64)
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
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H,W

class reSunTransformer1015(nn.Module):
    def __init__(self, img_size=224, patch_size=16, classes=2, channels=3,embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0,CNNdrop_rate=0.2,pretrained = True):
        super().__init__()
        self.num_classes = classes
        self.depths = depths
        self.num_stages = num_stages
        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=256, ch_out=256, drop_rate=CNNdrop_rate / 2)
        self.up_c_1 = Up(in_ch1=512, out_ch=256, in_ch2=256)
        self.up_c_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
        self.up_c_3 = Up(in_ch1=128, out_ch=64, in_ch2=64)
        self.up_t_1 = Up(in_ch1=512, out_ch=256)
        self.up_t_2 = Up(in_ch1=256, out_ch=128)
        self.up_t_3 = Up(in_ch1=128, out_ch=64)
        # deep supervision
        self.dsv5 = UnetDsv3(in_size=512, out_size=16, scale_factor=(48, 64))
        self.dsv4 = UnetDsv3(in_size=256, out_size=16, scale_factor=(48, 64))
        self.dsv3 = UnetDsv3(in_size=128, out_size=16, scale_factor=(48, 64))
        self.dsv2 = UnetDsv3(in_size=64, out_size=16, scale_factor=(48, 64))

        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(16)
        # nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=12, dilation=12)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv2 = conv3x3(16, 16)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.final_x = nn.Sequential(
            Conv(512, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )


        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('E:/fenge/TransFuse-1/pretrained2/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.drop = nn.Dropout2d(CNNdrop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=channels if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], classes) if classes > 0 else nn.Identity()

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != 4:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                d12 = x
                outs.append(d12)
        t = outs[3]
        t1 = outs[2]
        t2 = outs[1]
        t3 = outs[0]
        return t,t1,t2,t3


    def forward(self, x):
        c1=x
        x_t,x_t_1,x_t_2,x_t_3= self.forward_features(x)
        #cnn
        x_u = self.resnet.conv1(c1)  # (通道由3变64，大小变为1/2)     #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # 大小变为1/2（初始的1/4/）

        x_u_3 = self.resnet.layer1(x_u)  # self.layer1 = self._make_layer(block, 64, layers[0]
        x_u_3 = self.drop(x_u_3)

        x_u_2 = self.resnet.layer2(x_u_3)  # 通道加倍，高宽减半（初始的1/8）
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer3(x_u_2)  # 通道加倍，高宽减半（初始的1/16）
        x_u_1 = self.drop(x_u_1)

        x_u = self.resnet.layer4(x_u_1)  # 通道加倍，高宽减半（初始的1/32）
        x_u = self.drop(x_u)

        x_c = self.fusion_c(x_u, x_t)   #512,1/32

        x_c_1 = self.up_c_1(x_c, x_u_1)  #256,1/16
        x_c_2 = self.up_c_2(x_c_1, x_u_2) #128,1/8

        x_c_3 = self.up_c_3(x_c_2, x_u_3)  #64,1/4

        dsv5 = self.dsv5(x_c)    #512
        dsv4 = self.dsv4(x_c_1)  #256-16,1/16
        dsv3 = self.dsv3(x_c_2)  #128-16,1/8
        dsv2 = self.dsv2(x_c_3)  #64-16,1/4
        ds5= dsv5
        ds4=self.relu(self.bn4(self.conv4(dsv4)))
        ds3=self.relu(self.bn3(self.conv3(ds4+dsv3)))
        ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        ds5 = self.drop(ds5)
        ds4 = self.drop(ds4)
        ds3= self.drop(ds3)
        ds2 = self.drop(ds2)
        dsv_cat = torch.cat([ds2, ds3, ds4,ds5], dim=1)
        map_2 = F.interpolate(self.final_2(x_c_3), scale_factor=4, mode='bilinear')
        #map_2 = F.interpolate(self.final_2(dsv_cat), scale_factor=4, mode='bilinear')

        return torch.sigmoid(map_2)

class reSunTransformer1016(nn.Module):
    def __init__(self, img_size=224, patch_size=16, classes=2, channels=3,embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0,CNNdrop_rate=0.2,pretrained = True):
        super().__init__()
        self.num_classes = classes
        self.depths = depths
        self.num_stages = num_stages
        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=256, ch_out=256, drop_rate=CNNdrop_rate / 2)
        self.up_c_1 = Up(in_ch1=512, out_ch=256, in_ch2=256)
        self.up_c_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
        self.up_c_3 = Up(in_ch1=128, out_ch=64, in_ch2=64)
        self.up_t_1 = Up(in_ch1=512, out_ch=256)
        self.up_t_2 = Up(in_ch1=256, out_ch=128)
        self.up_t_3 = Up(in_ch1=128, out_ch=64)
        # deep supervision
        self.dsv5 = UnetDsv3(in_size=512, out_size=16, scale_factor=(48, 64))
        self.dsv4 = UnetDsv3(in_size=256, out_size=16, scale_factor=(48, 64))
        self.dsv3 = UnetDsv3(in_size=128, out_size=16, scale_factor=(48, 64))
        self.dsv2 = UnetDsv3(in_size=64, out_size=16, scale_factor=(48, 64))

        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(16)
        # nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=12, dilation=12)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv2 = conv3x3(16, 16)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.final_x = nn.Sequential(
            Conv(512, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )


        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('E:/fenge/TransFuse-1/pretrained2/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.drop = nn.Dropout2d(CNNdrop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=channels if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], classes) if classes > 0 else nn.Identity()

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != 4:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                d12 = x
                outs.append(d12)
        t = outs[3]
        t1 = outs[2]
        t2 = outs[1]
        t3 = outs[0]
        return t,t1,t2,t3


    def forward(self, x):
        c1=x
        x_t,x_t_1,x_t_2,x_t_3= self.forward_features(x)
        #cnn
        x_u = self.resnet.conv1(c1)  # (通道由3变64，大小变为1/2)     #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # 大小变为1/2（初始的1/4/）

        x_u_3 = self.resnet.layer1(x_u)  # self.layer1 = self._make_layer(block, 64, layers[0]
        x_u_3 = self.drop(x_u_3)

        x_u_2 = self.resnet.layer2(x_u_3)  # 通道加倍，高宽减半（初始的1/8）
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer3(x_u_2)  # 通道加倍，高宽减半（初始的1/16）
        x_u_1 = self.drop(x_u_1)

        x_u = self.resnet.layer4(x_u_1)  # 通道加倍，高宽减半（初始的1/32）
        x_u = self.drop(x_u)

        x_c = self.fusion_c(x_u, x_t)   #512,1/32

        x_c_1 = self.up_c_1(x_c, x_u_1)  #256,1/16
        x_c_2 = self.up_c_2(x_c_1, x_u_2) #128,1/8

        x_c_3 = self.up_c_3(x_c_2, x_u_3)  #64,1/4

        dsv5 = self.dsv5(x_c)    #512
        dsv4 = self.dsv4(x_c_1)  #256-16,1/16
        dsv3 = self.dsv3(x_c_2)  #128-16,1/8
        dsv2 = self.dsv2(x_c_3)  #64-16,1/4
        ds5= dsv5
        ds4=self.relu(self.bn4(self.conv4(dsv4)))
        ds3=self.relu(self.bn3(self.conv3(ds4+dsv3)))
        ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        ds5 = self.drop(ds5)
        ds4 = self.drop(ds4)
        ds3= self.drop(ds3)
        ds2 = self.drop(ds2)
        dsv_cat = torch.cat([ds2, ds3, ds4,ds5], dim=1)
        map_2 = F.interpolate(self.final_2(x_c_3), scale_factor=4, mode='bilinear')

        return torch.sigmoid(map_2)

class reSunTransformer1017(nn.Module):
    def __init__(self, img_size=224, patch_size=16, classes=2, channels=3,embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0,CNNdrop_rate=0.2,pretrained = True):
        super().__init__()
        self.num_classes = classes
        self.depths = depths
        self.num_stages = num_stages
        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=256, ch_out=256, drop_rate=CNNdrop_rate / 2)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=128, ch_out=128, drop_rate=CNNdrop_rate / 2)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=64, ch_out=64, drop_rate=CNNdrop_rate / 2)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64, drop_rate=CNNdrop_rate / 2)
        self.up_c_1 = Up(in_ch1=512, out_ch=256, in_ch2=256)
        self.up_c_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
        self.up_c_3 = Up(in_ch1=128, out_ch=64, in_ch2=64)
        self.up_t_1 = Up(in_ch1=512, out_ch=256)
        self.up_t_2 = Up(in_ch1=256, out_ch=128)
        self.up_t_3 = Up(in_ch1=128, out_ch=64)
        # deep supervision
        self.dsv5 = UnetDsv3(in_size=512, out_size=16, scale_factor=(48, 64))
        self.dsv4 = UnetDsv3(in_size=256, out_size=16, scale_factor=(48, 64))
        self.dsv3 = UnetDsv3(in_size=128, out_size=16, scale_factor=(48, 64))
        self.dsv2 = UnetDsv3(in_size=64, out_size=16, scale_factor=(48, 64))

        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(16)
        # nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=12, dilation=12)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv2 = conv3x3(16, 16)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.final_x = nn.Sequential(
            Conv(512, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )


        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('E:/fenge/TransFuse-1/pretrained2/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.drop = nn.Dropout2d(CNNdrop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=channels if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], classes) if classes > 0 else nn.Identity()

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != 4:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                d12 = x
                outs.append(d12)
        t = outs[3]
        t1 = outs[2]
        t2 = outs[1]
        t3 = outs[0]
        return t,t1,t2,t3


    def forward(self, x):
        c1=x
        x_t,x_t_1,x_t_2,x_t_3= self.forward_features(x)
        #cnn
        x_u = self.resnet.conv1(c1)  # (通道由3变64，大小变为1/2)     #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)  # 大小变为1/2（初始的1/4/）

        x_u_3 = self.resnet.layer1(x_u)  # self.layer1 = self._make_layer(block, 64, layers[0]
        x_u_3 = self.drop(x_u_3)

        x_u_2 = self.resnet.layer2(x_u_3)  # 通道加倍，高宽减半（初始的1/8）
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer3(x_u_2)  # 通道加倍，高宽减半（初始的1/16）
        x_u_1 = self.drop(x_u_1)

        x_u = self.resnet.layer4(x_u_1)  # 通道加倍，高宽减半（初始的1/32）
        x_u = self.drop(x_u)

        x_c = self.fusion_c(x_u, x_t)   #512,1/32
        x_c1 = self.fusion_c1(x_u_1, x_t_1)
        x_c2 = self.fusion_c2(x_u_2, x_t_2)
        x_c3 = self.fusion_c3(x_u_3, x_t_3)

        # x_c_1 = self.up_c_1(x_c, x_u_1)  #256,1/16
        x_c_1 = self.up_c_1(x_c, x_c1)  # 256,1/16
        x_c_2 = self.up_c_2(x_c_1, x_c2) #128,1/8

        x_c_3 = self.up_c_3(x_c_2, x_c3)  #64,1/4

        dsv5 = self.dsv5(x_c)    #512
        dsv4 = self.dsv4(x_c_1)  #256-16,1/16
        dsv3 = self.dsv3(x_c_2)  #128-16,1/8
        dsv2 = self.dsv2(x_c_3)  #64-16,1/4
        ds5= dsv5
        ds4=self.relu(self.bn4(self.conv4(dsv4)))
        ds3=self.relu(self.bn3(self.conv3(ds4+dsv3)))
        ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        ds5 = self.drop(ds5)
        ds4 = self.drop(ds4)
        ds3= self.drop(ds3)
        ds2 = self.drop(ds2)
        dsv_cat = torch.cat([ds2, ds3, ds4,ds5], dim=1)
        map_2 = F.interpolate(self.final_2(x_c_3), scale_factor=4, mode='bilinear')

        return torch.sigmoid(map_2)

class reSunTransformer1018(nn.Module):
    def __init__(self, img_size=224, patch_size=16, classes=2, channels=3,embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0,CNNdrop_rate=0.2,pretrained = True):
        super().__init__()
        self.num_classes = classes
        self.depths = depths
        self.num_stages = num_stages
        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=256, ch_out=256, drop_rate=CNNdrop_rate / 2)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=128, ch_out=128, drop_rate=CNNdrop_rate / 2)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=64, ch_out=64, drop_rate=CNNdrop_rate / 2)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64, drop_rate=CNNdrop_rate / 2)
        self.up_c_1 = Up(in_ch1=512, out_ch=256, in_ch2=256)
        self.up_c_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
        self.up_c_3 = Up(in_ch1=128, out_ch=64, in_ch2=64)
        self.up_t_1 = Up(in_ch1=512, out_ch=256)
        self.up_t_2 = Up(in_ch1=256, out_ch=128)
        self.up_t_3 = Up(in_ch1=128, out_ch=64)
        # deep supervision
        self.dsv5 = UnetDsv3(in_size=512, out_size=64, scale_factor=(56, 80))
        self.dsv4 = UnetDsv3(in_size=256, out_size=64, scale_factor=(56, 80))
        self.dsv3 = UnetDsv3(in_size=128, out_size=64, scale_factor=(56, 80))
        self.dsv2 = UnetDsv3(in_size=64, out_size=64, scale_factor=(56, 80))

        # self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=6, dilation=6)
        # self.bn3 = nn.BatchNorm2d(64)
        # nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        # self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=12, dilation=12)
        # self.bn4 = nn.BatchNorm2d(64)

        # self.conv5 = nn.Conv2d(64, 64, 3, 1, padding=18, dilation=18)
        # self.bn5 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(384, 128, 3, 1, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(768, 256, 3, 1, padding=12, dilation=12)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(512, 512, 3, 1, padding=18, dilation=18)
        self.bn5 = nn.BatchNorm2d(512)

        self.upup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv2 = conv3x3(192, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.final_x = nn.Sequential(
            Conv(512, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )


        self.final_2 = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )

        # self.resnet = resnet()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('E:/fenge/TransFuse-1/pretrained2/resnet34-43635321.pth'))
        # self.resnet.fc = nn.Identity()
        self.drop = nn.Dropout2d(CNNdrop_rate)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.fc1 = nn.Conv2d(64, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, 64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.attentiongate1=Attention_block(F_g=512, F_l=512, F_int=256)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=channels if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], classes) if classes > 0 else nn.Identity()

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != 4:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                d12 = x
                outs.append(d12)
        t = outs[3]
        t1 = outs[2]
        t2 = outs[1]
        t3 = outs[0]
        return t,t1,t2,t3


    def forward(self, x):
        c1=x
        x_t,x_t_1,x_t_2,x_t_3= self.forward_features(x)
        #cnn
        # x_u = self.resnet.conv1(c1)  # (通道由3变64，大小变为1/2)     #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # x_u = self.resnet.bn1(x_u)
        # x_u = self.resnet.relu(x_u)
        # x_u = self.resnet.maxpool(x_u)  # 大小变为1/2（初始的1/4/）
        #
        # x_u_3 = self.resnet.layer1(x_u)  # self.layer1 = self._make_layer(block, 64, layers[0]
        # x_u_3 = self.drop(x_u_3)
        #
        # x_u_2 = self.resnet.layer2(x_u_3)  # 通道加倍，高宽减半（初始的1/8）
        # x_u_2 = self.drop(x_u_2)
        #
        # x_u_1 = self.resnet.layer3(x_u_2)  # 通道加倍，高宽减半（初始的1/16）
        # x_u_1 = self.drop(x_u_1)
        #
        # x_u = self.resnet.layer4(x_u_1)  # 通道加倍，高宽减半（初始的1/32）
        # x_u = self.drop(x_u)
        # Encoder
        x = self.firstconv(x)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(x)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]

        # x_c = self.fusion_c(x_u, x_t)   #512,1/32
        # x_c1 = self.fusion_c1(x_u_1, x_t_1)
        # x_c2 = self.fusion_c2(x_u_2, x_t_2)
        # x_c3 = self.fusion_c3(x_u_3, x_t_3)
        #
        # # x_c_1 = self.up_c_1(x_c, x_u_1)  #256,1/16
        # x_c_1 = self.up_c_1(x_c, x_c1)  # 256,1/16
        # x_c_2 = self.up_c_2(x_c_1, x_c2) #128,1/8
        #
        # x_c_3 = self.up_c_3(x_c_2, x_c3)  #64,1/4
        x_c = self.fusion_c(x_u, x_t)  # 512,1/32

        x_c_1 = self.up_c_1(x_c, x_u_1)  # 256,1/16
        x_c_2 = self.up_c_2(x_c_1, x_u_2)  # 128,1/8

        x_c_3 = self.up_c_3(x_c_2, x_u_3)  # 64,1/4

        # dsv5 = self.dsv5(x_c)    #512
        # dsv4 = self.dsv4(x_c_1)  #256-16,1/16
        # dsv3 = self.dsv3(x_c_2)  #128-16,1/8
        # dsv2 = x_c_3             #64-16,1/4   #64-64,1/4

        dsv5 = x_c # 512
        dsv4 = x_c_1  # 256-16,1/16
        dsv3 = x_c_2  # 128-16,1/8
        dsv2 = x_c_3  # 64-16,1/4   #64-64,1/4
        ds5 = self.relu(self.bn5(self.conv5(dsv5)))   #512/512
        ds5=self.upup(ds5)

        ds4 = self.relu(self.bn4(self.conv4(torch.cat([ds5, dsv4], dim=1))))  #768/256
        ds4 = self.upup(ds4)


        ds3 = self.relu(self.bn3(self.conv3(torch.cat([ds4, dsv3], dim=1))))   #384/128
        ds3 = self.upup(ds3)

        ds2 = self.relu(self.bn2(self.conv2(torch.cat([ds3, dsv2], dim=1))))   #192/64




        # dsv5 = self.dsv5(x_c)  # 512     #512-16,1/32
        # dsv4 = self.dsv4(x_c_1)  #256
        # dsv3 = self.dsv3(x_c_2)
        # dsv2 = self.dsv2(x_c_3)
        #
        # ds5 = self.relu(self.bn5(self.conv5(dsv5)))
        # ds4 = self.relu(self.bn4(self.conv4(ds5 + dsv4)))
        # ds3 = self.relu(self.bn3(self.conv3(ds4 + dsv3)))
        # ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        # dsv_cat = torch.cat([ds2, ds3, ds4, ds5], dim=1)



        # # Deep Supervision
        # dsv4 = self.dsv4(x_c)  # up4:128,1/8  #256-22,1/16
        # dsv3 = self.dsv3(x_c_1)  # up3:64,1/4   #128-21,1/8
        # dsv2 = self.dsv2(x_c_2)  # up2:32,1/2   #64-22,1/4
        #
        # ds4 = self.relu(self.bn4(self.conv4(dsv4)))
        # ds3 = self.relu(self.bn3(self.conv3(ds4 + dsv3)))
        # ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        #
        # dsv_cat = torch.cat([ds2, ds3, ds4,ds5], dim=1)

        #map_2 = F.interpolate(self.final_1(x_c_3), scale_factor=4, mode='bilinear')
        map_2 = F.interpolate(self.final_1(x_c_3), scale_factor=4, mode='bilinear')

        return torch.sigmoid(map_2)
class reSunTransformer1019(nn.Module):
    def __init__(self, img_size=224, patch_size=16, classes=2, channels=3,embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0,CNNdrop_rate=0.2,pretrained = True):
        super().__init__()
        self.num_classes = classes
        self.depths = depths
        self.num_stages = num_stages
        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=256, ch_out=256, drop_rate=CNNdrop_rate / 2)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=128, ch_out=128, drop_rate=CNNdrop_rate / 2)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=64, ch_out=64, drop_rate=CNNdrop_rate / 2)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=64, ch_out=64, drop_rate=CNNdrop_rate / 2)

        self.final_x = nn.Sequential(
            Conv(512, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )


        self.final_2 = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )

        # self.resnet = resnet()
        # if pretrained:
        #     self.resnet.load_state_dict(torch.load('E:/fenge/TransFuse-1/pretrained2/resnet34-43635321.pth'))
        # self.resnet.fc = nn.Identity()
        self.drop = nn.Dropout2d(CNNdrop_rate)

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
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

        # deep supervision
        self.dsv5 = UnetDsv3(in_size=512, out_size=16, scale_factor=(56, 80))
        self.dsv4 = UnetDsv3(in_size=256, out_size=16, scale_factor=(56, 80))
        self.dsv3 = UnetDsv3(in_size=128, out_size=16, scale_factor=(56, 80))
        self.dsv2 = UnetDsv3(in_size=64, out_size=16, scale_factor=(56, 80))

        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(16)
        # nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=12, dilation=12)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv2 = conv3x3(16, 16)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(64, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, 64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.attentiongate1=Attention_block(F_g=512, F_l=512, F_int=256)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=channels if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], classes) if classes > 0 else nn.Identity()

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != 4:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                d12 = x
                outs.append(d12)
        t = outs[3]
        t1 = outs[2]
        t2 = outs[1]
        t3 = outs[0]
        return t,t1,t2,t3


    def forward(self, x):
        c1=x
        x_t,x_t_1,x_t_2,x_t_3= self.forward_features(x)
        #cnn
        # Encoder
        x = self.firstconv(x)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(x)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]

        x_c = self.fusion_c(x_u, x_t)  # 512,1/32
        d4 = self.decoder4(x_c) + x_u_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d3 = self.decoder3(d4) + x_u_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d2 = self.decoder2(d3) + x_u_3  # [2, 64, 56, 80]
        d1 = self.decoder1(d2)  # [2, 64, 112, 160]

        dsv5 = self.dsv5(x_c)  # 512
        dsv4 = self.dsv4(x_c_1)  # 256-16,1/16
        dsv3 = self.dsv3(x_c_2)  # 128-16,1/8
        dsv2 = self.dsv2(x_c_3)  # 64-16,1/4

        ds5 = dsv5
        ds4 = self.relu(self.bn4(self.conv4(dsv4)))
        ds3 = self.relu(self.bn3(self.conv3(ds4 + dsv3)))
        ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        ds5 = self.drop(ds5)
        ds4 = self.drop(ds4)
        ds3 = self.drop(ds3)
        ds2 = self.drop(ds2)
        dsv_cat = torch.cat([ds2, ds3, ds4, ds5], dim=1)

        # map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear', align_corners=True)
        # map_1 = F.interpolate(self.final_1(x_t_3), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(dsv_cat), scale_factor=4, mode='bilinear')

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        return torch.sigmoid(out)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
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

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


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
class DoubleViT(nn.Module):
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

        self.sa4 = SqueezeAttentionBlock(ch_in=256, ch_out=256)
        self.sa3 = SqueezeAttentionBlock(ch_in=128, ch_out=128)
        self.sa2 = SqueezeAttentionBlock(ch_in=64, ch_out=64)
        self.sa1 = SqueezeAttentionBlock(ch_in=64, ch_out=64)

        self.Attentiongate1 = AttentionBlock(256, 256, 256)
        self.Attentiongate2 = AttentionBlock(128, 128, 128)
        self.Attentiongate3 = AttentionBlock(64, 64, 64)

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=768, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=192, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=96, r_2=4, ch_int=64, ch_out=64)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv112 = nn.ConvTranspose2d(80, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)
        self.se=SEBlock(channel=64)

        self.dsv5 = UnetDsv3(in_size=512, out_size=16, scale_factor=(96, 128))
        self.dsv4 = UnetDsv3(in_size=256, out_size=16, scale_factor=(96, 128))
        self.dsv3 = UnetDsv3(in_size=128, out_size=16, scale_factor=(96, 128))
        self.dsv2 = UnetDsv3(in_size=64, out_size=16, scale_factor=(96, 128))
        self.dsv1 = UnetDsv3(in_size=64, out_size=16, scale_factor=(96, 128))



        self.conv3 = nn.Conv2d(16, 16, 3, 1, padding=6, dilation=6)
        self.bn3 = nn.BatchNorm2d(16)
        # nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=12, dilation=12)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv2 = conv3x3(16, 16)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv1 = conv3x3(16, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 2, 3, bn=False, relu=False)
        )

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
        c0=x
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
            o1=out
            if i==0:
                c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
                c1 = self.firstbn(c1)
                c1 = self.firstrelu(c1)
                c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
                c1 = self.encoder1(c1)
                fuse1 = self.fusion_c3(c1, out)  # 64
                out=fuse1
                soter1=fuse1
            elif i == 1:
                c1 = self.encoder2(c1)
                fuse2 = self.fusion_c2(c1, out)
                out = fuse2
                soter2 = fuse2
            elif i == 2:
                c1 = self.encoder3(c1)
                fuse3 = self.fusion_c1(c1, out)
                out = fuse3
                soter3 = fuse3
            elif i == 3:
                c1 = self.encoder4(c1)
                out = self.fusion_c(c1, out)
                soter4=out


        x_f = soter4
        x_f_1 = soter3
        x_f_2 = soter2 # 192
        x_f_3 = soter1


        d4 = self.decoder4(x_f) + x_f_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        # d4=self.sa4(d4)
        d3 = self.decoder3(d4) + x_f_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        # d3 = self.sa3(d3)
        d2 = self.decoder2(d3) + x_f_3  # [2, 64, 56, 80]
        # d2 = self.sa2(d2)
        d1 = self.decoder1(d2)  # [2, 64, 112, 160]
        # d1 = self.sa1(d1)

        # dsv5 = self.dsv5(x_f)  # 512
        # dsv4 = self.dsv4(d4)  # 256-16,1/16
        # dsv3 = self.dsv3(d3)  # 128-16,1/8
        # dsv2 = self.dsv2(d2)
        # dsv1 = self.dsv1(d1) # 64-16,1/4
        #
        #
        # ds5 = dsv5
        # ds4 = self.relu(self.bn4(self.conv4(dsv4)))
        # ds3 = self.relu(self.bn3(self.conv3(ds4 + dsv3)))
        # ds2 = self.relu(self.bn2(self.conv2(ds3 + dsv2)))
        # ds1 = self.relu(self.bn1(self.conv1(ds2 + dsv1)))
        #
        # dsv_cat = torch.cat([ds1,ds2, ds3, ds4, ds5], dim=1)

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        # out = self.finaldeconv112(dsv_cat)  # [2, 32, 224, 320]
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)  # [2, 32, 224, 320]
        # out = self.finalrelu2(out)
        # out = self.finalconv3(out)  # [2, 2, 224, 320]




        return out

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DoubleViT, self).train(mode)



if __name__ == '__main__':
    input = torch.rand(2, 3, 64, 64)


    # model = RCEM(3, 64)
    # model = BasicBlock(3, 64, 64)
    model = DaViT()
    #model = Feature_Pyramid_Fusion(high_feature=a, inner_featurea=b, low_feature=c)
    out12 = model(input)
    print(out12.shape)