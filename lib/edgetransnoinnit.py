import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
from DeiT import deit_small_patch16_224 as deit
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.resnet import BasicBlock as ResBlock
from models import GSConv as gsc
from models.norm import Norm2d

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)  # //除完取整
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)  # 卷积
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)  # 乘以（中间的）

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch   #（x）
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))  # x是 trans分支，g是cnn分支，，bp是中间的

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class edgTransFuse_S(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):  #
        super(edgTransFuse_S, self).__init__()

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = deit(pretrained=pretrained)



        self.up1 = Up(in_ch1=384, out_ch=128)  #
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(  # （256，64）（64，64） （64，numclass）
            Conv(256, 64, 1, bn=True, relu=True),  # （64，64） （64，numclass）
            Conv(64, 64, 3, bn=True, relu=True),  # （64，64） （64，numclass）
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)  #

        self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate / 2)  #
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.drop = nn.Dropout2d(drop_rate)
        # Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)  # (128,1)
        self.c4 = nn.Conv2d(512, 1, kernel_size=1)  # (256,1)

        self.d0 = nn.Conv2d(128, 64, kernel_size=1)  # (64,32)
        self.res1 = ResBlock(64, 64)  # (32,32)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)  # (32,16)
        self.res2 = ResBlock(32, 32)  # (16,16)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)  # (16,8)
        self.gate1 = gsc.GatedSpatialConv2d(16, 16)
        self.gate2 = gsc.GatedSpatialConv2d(8, 8)

      #  if normal_init:
       #     self.init_weights()

def forward(self, imgs, labels=None):
        # bottom-up path
        x_b = self.transformer(imgs)  # 应该是下采样结果H/16，W/16
        f1=x_b

        x_b = torch.transpose(x_b, 1, 2)  # 交换torch（【第一，第二，第三】）（0.1.2）交换后两个的维度
        x_b = x_b.view(x_b.shape[0], -1, 12, 16)  # view就是reshape   shape[0]就是行
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)  # 输入384 输出128
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here      #输入128  输出64
        x_b_2 = self.drop(x_b_2)

        # top-down path
        x_u = self.resnet.conv1(
            imgs)  # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)  # self.layer1 = self._make_layer(block, 64, layers[0]
        x_u_2 = self.drop(x_u_2)

        x_u_1 = self.resnet.layer2(x_u_2)  #
        x_u_1 = self.drop(x_u_1)

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)

        # joint path
        x_c = self.up_c(x_u, x_b)  # BiFision        xb应该是1/16(trans)      xu cnn                                                     [256,256]

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)  # BiFision   1/8的结果                                                                      [128,128]
        x_c_1 = self.up_c_1_2(x_c,
                              x_c_1_1)  # 1/16fision的上采样+1/8通过注意力在相加，再卷积    (1/8的结果的fusion结果xc11与)  Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)                                                       #bifusion                             [64,64]
        x_c_2 = self.up_c_2_2(x_c_1,
                              x_c_2_1)  # joint predict low supervise here     #self.up_c_2_2 = Up(128, 输出64, 64, attn=True)

        #shape
        # Shape Stream
        x_size = imgs.size()
        ss = F.interpolate(self.d0(x_c_2_1), x_size[2:],    mode='bilinear', align_corners=True)
        ss = self.res1(ss)                                 #保持不变   [32,32]
        c3 = F.interpolate(self.c3(x_c_1_1), x_size[2:],    mode='bilinear', align_corners=True)
                           #[64.128]  c3:[128,1]变1
        ss = self.d1(ss)                                   #[32,16]
        ss = self.gate1(ss, c3)                      #self.gate1 = gsc.GatedSpatialConv2d(16, 16)

        ss = self.res2(ss)                                 # [16,16]
        ss = self.d2(ss)                                   #[16,8]
        c4 = F.interpolate(self.c4(x_c), x_size[2:],    mode='bilinear', align_corners=True)         #self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        ss = self.gate2(ss, c4)                         #self.gate2 = gsc.GatedSpatialConv2d(8, 8)
                                  # ss = self.res3(ss)                              self.res3 = ResBlock(8, 8)
        ss = self.fuse(ss)                            # self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)           #edge loss

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16,
                              mode='bilinear')  # （256，64）（64，64） （64，1） #cnn trans分支，混合分支
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear')  # （64，64） （64，numclass）    #trans分支
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear')  # 最后融合

        print('out第一个', map_2.size(0))
        print('第二个', map_2.size(1))
        print('第三个', map_2.size(2))
        print('第三个', map_2.size(3))

        return map_x, map_1, map_2, edge_out



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

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

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
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