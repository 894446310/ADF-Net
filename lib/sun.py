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
from models.layers.modules import UnetDsv3

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

class Up(nn.Module):
    """Upscaling then double conv"""   #Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)   #in,out

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
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

class h_swish(nn.Module):
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x
class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()
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

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_2, ch_2)
        self.conv_atten = conv_block(ch_2, ch_2)
        # self.upsample = nn.Upsample(size=(6,8), mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2)

        # bi-linear modelling for both
        self.cnn = add_conv(ch_1, ch_1, 1, 1)
        self.trans = add_conv(ch_1, ch_1, 1, 1)
        self.weight1 = add_conv(ch_1, 16, 1, 1)
        self.weight2 = add_conv(ch_1, 16, 1, 1)
        self.weight_trans = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)

        #attention
        self.Adap_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, 512, 1, 1))
        self.cbias = Parameter(torch.ones(1, 512, 1, 1))
        self.sweight = Parameter(torch.zeros(1, 512, 1, 1))
        self.sbias = Parameter(torch.ones(1, 512, 1, 1))
        self.gn = nn.GroupNorm(512, 512)

        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        self.inplanes = 512
        # self.F1 = nn.Sequential(
        #     nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, dilation=dilation[0], padding=dilation[0]),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU()
        # )
        # self.F2 = nn.Sequential(
        #     nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, dilation=dilation[1], padding=dilation[1]),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU()
        # )
        self.Conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1)

        r = 8
        L = 48
        d = max(int(self.inplanes / r), L)
        self.W1 = nn.Linear(2 * self.inplanes, d)
        self.W2 = nn.Linear(d, self.inplanes)
        self.W3 = nn.Linear(d, self.inplanes)

    def forward(self, g, x):
        U1 = g
        U2 = x
        fuse = g + x

        # fuse = U1 * a1 + U2 * a2
        # avg_pool = F.avg_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        # max_pool = F.max_pool2d(U, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        # p = torch.mean(torch.mean(U, dim=2), dim=2).reshape(-1, self.inplanes, 1, 1)
        # q = torch.matmul(
        #     U.reshape([-1, self.inplanes, x.shape[2] * x.shape[3]]),
        #     (nn.Softmax(dim=1)(self.Conv_q(x))).reshape([-1, x.shape[2] * x.shape[3], 1])
        # ).reshape([-1, self.inplanes, 1, 1])
        # sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
        # s = torch.sigmoid(torch.cat([p, q], 1)).reshape([-1, self.inplanes * 2])
        # z1 = self.W2(nn.ReLU()(self.W1(s)))
        # z2 = self.W3(nn.ReLU()(self.W1(s)))
        # zc1 = self.W2(nn.ReLU()(self.W1(sc)))
        # zc2 = self.W3(nn.ReLU()(self.W1(sc)))
        # a1 = (torch.exp(z1) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
        # a2 = (torch.exp(z2) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
        # ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        # ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        # fuse = U1 * a1 + U2 * a2 + U
        # bilinear pooling  g是cnn，256  x是transformer，384
        #W_g = self.cnn(g)  # 卷积Conv(ch_1, ch_int, 1, bn=True, relu=False)  (256，256)
        #W_x = self.W_x(x) # (384,256)
        #W_x =self.trans(x)
        # W_x = self.cnn(g)
        #bp = self.W(W_g * W_x)  # 乘以（中间的）  (256,256)


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
        # s0=yx
        # yx = self.conv_atten(yx)
        # s1=yx
        # yx = self.upsample(yx)
        # x=(yx * x_res) + yx
        # xn = self.Adap_pool(x)
        # xn = self.cweight * xn + self.cbias
        # xn = x * self.sigmoid(xn)
        # W_x=xn
        #
        # gs = self.gn(g)
        # gs = self.sweight * gs + self.sbias
        # W_g = g * self.sigmoid(gs)
        #
        #
        # # W_g = self.cnn(g)
        # # W_x = self.trans(x)
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

class  SunTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0,CNNdrop_rate=0.2,pretrained = True):
        super().__init__()
        self.num_classes = num_classes
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
            Conv(64, 1, 3, bn=False, relu=False)
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
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
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
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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

    # def forward_features(self, x):
    #     B = x.shape[0]
    #     outs=[]
    #     outs2=[]
    #     o1=x
    #     #self.num_stages
    #
    #     for i in range(0,4):
    #         patch_embed = getattr(self, f"patch_embed{i + 1}")
    #         block = getattr(self, f"block{i + 1}")
    #         norm = getattr(self, f"norm{i + 1}")
    #         x, H, W = patch_embed(x)
    #         f0=x
    #         for blk in block:
    #             x = blk(x, H, W)
    #         n0=x
    #         x = norm(x)
    #         n1=x
    #         n2=self.num_stages - 1
    #         # if i != self.num_stages - 1:
    #         #     d11=x
    #         #     outs.append(d11)
    #         #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    #         #     d12=x
    #         #     outs2.append(d12)
    #         #if i != self.num_stages - 1:
    #         if i != 3:
    #             d11=x
    #             outs.append(d11)
    #             x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    #             d12=x
    #             outs2.append(d12)
    #     #return x.mean(dim=1)
    #     d13=x
    #     return x
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
        # x_t=x
        # x_t_1=self.up_t_1(x_t)
        # x_t_1 = self.drop(x_t_1)
        #
        # x_t_2 = self.up_t_2(x_t_1)
        # x_t_2 = self.drop(x_t_2)
        #
        # x_t_3 = self.up_t_3(x_t_2)
        # x_t_3 = self.drop(x_t_3)

        #cnn
        x_u = self.resnet.conv1(c1)  # (通道由3变64，大小变为1/2)     #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        e30 = x_u
        x_u = self.resnet.bn1(x_u)
        e31 = x_u
        x_u = self.resnet.relu(x_u)
        e32 = x_u
        x_u = self.resnet.maxpool(x_u)  # 大小变为1/2（初始的1/4/）
        e4 = x_u

        x_u_3 = self.resnet.layer1(x_u)  # self.layer1 = self._make_layer(block, 64, layers[0]
        x_u_3 = self.drop(x_u_3)

        x_u_2 = self.resnet.layer2(x_u_3)  # 通道加倍，高宽减半（初始的1/8）
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer3(x_u_2)  # 通道加倍，高宽减半（初始的1/16）
        x_u_1 = self.drop(x_u_1)

        x_u = self.resnet.layer4(x_u_1)  # 通道加倍，高宽减半（初始的1/32）
        x_u = self.drop(x_u)

        x_c = self.fusion_c(x_u, x_t)   #512,1/32
        #x_c11=self.fusion_c1(x_u_1, x_t_1)

        x_c_1 = self.up_c_1(x_c, x_u_1)  #256,1/16

        #x_c21 = self.fusion_c2(x_u_2, x_t_2)
        x_c_2 = self.up_c_2(x_c_1, x_u_2) #128,1/8

        #x_c31 = self.fusion_c3(x_u_3, x_t_3)
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

        # map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear', align_corners=True)
        # map_1 = F.interpolate(self.final_1(x_t_3), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_3), scale_factor=4, mode='bilinear')
        mm=F.sigmoid(map_2)

        return map_2



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


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

# def main():
#     print(torch.cuda.is_available())
#     print(torch.version.cuda)
#     print(torch.__version__)
#     input = torch.rand((2, 3, 224, 224))
#     model = SunTransformer()
#     output = model(input)
#     print(model)
#     print(output.size())  # [B, H/8 * W/8, 2*embed_dim]
#     model = SunTransformer().cuda
#     batch_size = 2
#     summary(model, input_size=(3, 224, 224))
#
#
#
# if __name__ == '__main__':
#     main()
# #打印网络
# import torchinfo
# from torchinfo import summary
# from lib.swinsiwnreal import SwinTransformer
# from lib.unet import U_Net
# model = SunTransformer()
# batch_size = 2
# summary(model, input_size=(batch_size, 3, 448, 448))
#
# model2 = SwinTransformer()
# batch_size = 2
# summary(model2, input_size=(batch_size, 3, 224, 224))