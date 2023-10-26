from functools import partial
import numpy as np
import torch.nn.functional as F

import torch
from torch import nn

from lib import pvt_v2
from timm.models.vision_transformer import _cfg
from torchvision import models
from functools import partial
from torchvision.models import resnet18 as resnet

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=352,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("D:/FCBFormer-main/FCBFormer-main/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64), nn.Upsample(size=88)
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

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
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l

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


# class BiFusion_block(nn.Module):  # BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
#     def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
#                  drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
#         super(BiFusion_block, self).__init__()
#
#         self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (256到256）
#         self.W_x = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (384，256)
#         self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
#         self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
#         self.dropout = nn.Dropout2d(drop_rate)
#         self.drop_rate = drop_rate
#         self.inplanes = ch_int
#         self.Conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1)
#         r = 8
#         L = 48
#         d = max(int(self.inplanes / r), L)
#         self.W1 = nn.Linear(2 * self.inplanes, d)
#         self.W2 = nn.Linear(d, self.inplanes)
#         self.W3 = nn.Linear(d, self.inplanes)
#
#     def forward(self, x, g):
#         U1 = self.W_g(g)
#         U2 = self.W_x(x)
#         U3=self.W(U1*U2)
#         #U = U1 + U2 + U3
#         U = self.residual(torch.cat([g, x, U3], 1))
#         avg_pool = F.avg_pool2d(U1, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
#         max_pool = F.max_pool2d(U2, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
#         sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
#         zc1 = self.W2(nn.ReLU()(self.W1(sc)))
#         zc2 = self.W3(nn.ReLU()(self.W1(sc)))
#         ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
#         ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
#         fuse = U1 * ac1 + U2 * ac2
#         return U1
class BiFusion_block(nn.Module):  # BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()

        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (256到256）
        self.W_x = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (384，256)

        self.relu = nn.ReLU(inplace=True)
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.inplanes = ch_int
        self.Conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1)
        r = 8
        L = 48
        d = max(int(self.inplanes / r), L)
        self.W1 = nn.Linear(2 * self.inplanes, d)
        self.W2 = nn.Linear(d, self.inplanes)
        self.W3 = nn.Linear(d, self.inplanes)

    def forward(self, x, g):
        U1 = self.W_g(g)
        U2 = self.W_x(x)
        U=U1+U2
        avg_pool = F.avg_pool2d(U1, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        max_pool = F.max_pool2d(U2, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
        zc1 = self.W2(nn.ReLU()(self.W1(sc)))
        zc2 = self.W3(nn.ReLU()(self.W1(sc)))
        ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        fuse = U1 * ac1 + U2 * ac2
        return fuse
nonlinearity = partial(F.relu, inplace=True)

class spAttention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, r_2):
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
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
class TA(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        checkpoint = torch.load("D:/PVT-2/PVT-2/pvt_v2_b0.pth")
        #checkpoint = torch.load("D:/FCBFormer-main/FCBFormer-main/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))


        resnet = models.resnet18(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        filters = [32, 64, 160, 256]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=4, ch_int=512, ch_out=512)
        # self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=160, r_2=4, ch_int=256, ch_out=256)
        # self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=64, r_2=4, ch_int=128, ch_out=128)
        # self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=32, r_2=4, ch_int=64, ch_out=64)
        self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=8, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=160, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=64, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate / 2)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=32, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate / 2)

        # self.up_c = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        # self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate / 2)
        # self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate / 2)

        self.attn_block1 = spAttention_block(256, 256, 256, 4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

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
        img=x
        pyramid = self.get_pyramid(x)
        x = self.firstconv(img)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        e1 = self.encoder1(x)     # [2, 64, 56, 80]
        e2 = self.encoder2(e1)    # [2, 128, 28, 40]
        e3 = self.encoder3(e2)    # [2, 256, 14, 20]
        e4 = self.encoder4(e3)    # [2, 512, 7, 10]
        # Decoder
        x_c = self.fusion_c(e4, pyramid[3])  # 512,1/32
        x_c_1 = self.fusion_c1(e3, pyramid[2])  # 256
        x_c_2 = self.fusion_c2(e2, pyramid[1])  # 128
        x_c_3 = self.fusion_c3(e1, pyramid[0])  # 64

        # d4 = self.decoder4(x_c) + x_c_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        # d3 = self.decoder3(d4) + x_c_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        # d2 = self.decoder2(d3)
        # d2 = self.decoder2(d3) + x_c_3  # [2, 64, 56, 80]
        # d1 = self.decoder1(d2)       # [2, 64, 112, 160]
        d40 = self.decoder4(x_c)
        d41= self.attn_block1(d40, x_c_1)
        d4 = d40 + d41
        d30 = self.decoder3(d4)
        d31 = self.attn_block2(d30, x_c_2)
        d3 = d30 + d31
        d20 = self.decoder2(d3)
        d21= self.attn_block3(d20, x_c_3)
        d2 = d20 + d21
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # [2, 2, 224, 320]

        return out
class TTC(nn.Module):
    def __init__(self):

        super().__init__()
        backbone = pvt_v2.PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
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

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=filters[3], ch_out=filters[3])
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=320, r_2=4, ch_int=filters[2], ch_out=filters[2])
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=128, r_2=4, ch_int=filters[1], ch_out=filters[1])
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=filters[0], ch_out=filters[0])

        self.attn_block1 = spAttention_block(filters[2], filters[2], filters[2], 4)
        self.attn_block2 = spAttention_block(filters[1], filters[1], filters[1], 4)
        self.attn_block3 = spAttention_block(filters[0], filters[0], filters[0], 4)


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
        img=x
        pyramid = self.get_pyramid(x)
        x = self.firstconv(img)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # [2, 64, 56, 80]
        e1 = self.encoder1(x)     # [2, 64, 56, 80]
        e2 = self.encoder2(e1)    # [2, 128, 28, 40]
        e3 = self.encoder3(e2)    # [2, 256, 14, 20]
        e4 = self.encoder4(e3)    # [2, 512, 7, 10]
        # Decoder
        x_c = self.fusion_c(e4, pyramid[3])  # 512,1/32
        x_c_1 = self.fusion_c1(e3, pyramid[2])  # 256
        x_c_2 = self.fusion_c2(e2, pyramid[1])  # 128
        x_c_3 = self.fusion_c3(e1, pyramid[0])  # 64

        d40 = self.decoder4(x_c)
        d41= self.attn_block1(d40, x_c_1)
        d4 = d40 + d41
        d30 = self.decoder3(d4)
        d31 = self.attn_block2(d30, x_c_2)
        d3 = d30 + d31
        d20 = self.decoder2(d3)
        d21= self.attn_block3(d20, x_c_3)
        d2 = d20 + d21
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # [2, 2, 224, 320]
        map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(d2), scale_factor=4, mode='bilinear', align_corners=True)

        # d2 = F.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)
        return map_x,map_1,out

class DoubleTC(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        checkpoint = torch.load("D:/PVT-2/PVT-2/pvt_v2_b0.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))
        resnet = models.resnet18(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        filters = [32, 64, 160, 256]
        # filters = [64, 128, 256, 512]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=4, ch_int=filters[3], ch_out=filters[3])
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=160, r_2=4, ch_int=filters[2], ch_out=filters[2])
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=64, r_2=4, ch_int=filters[1], ch_out=filters[1])
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=32, r_2=4, ch_int=filters[0], ch_out=filters[0])
        self.c=Conv(32, 64, 1, bn=True, relu=True)
        self.c1 = Conv(64, 128, 1, bn=True, relu=True)
        self.c2 = Conv(160, 256, 1, bn=True, relu=True)
        self.c3 = Conv(256, 512, 1, bn=True, relu=True)
        self.attn_block1 = spAttention_block(filters[2], filters[2], filters[2], 4)
        self.attn_block2 = spAttention_block(filters[1], filters[1], filters[1], 4)
        self.attn_block3 = spAttention_block(filters[0], filters[0], filters[0], 4)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        img=x
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)#bchw--bc,h*w
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                if i==2:
                    c1 = self.firstconv(img)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
                    c1 = self.firstbn(c1)
                    c1 = self.firstrelu(c1)
                    c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
                    c1 = self.encoder1(c1)
                    fuse1 = self.fusion_c3(c1, x)
                    c1=self.c(fuse1)
                    x = fuse1
                    soter1 = fuse1
                elif i == 5:
                    c1 = self.encoder2(c1)
                    fuse2 = self.fusion_c2(c1, x)
                    c1=self.c1(fuse2)
                    x = fuse2
                    soter2 = fuse2
                elif i == 8:
                    c1 = self.encoder3(c1)
                    fuse3 = self.fusion_c1(c1, x)
                    c1=self.c2(fuse3)
                    x = fuse3
                    soter3 = fuse3
                elif i == 11:
                    c1 = self.encoder4(c1)
                    fuse4 = self.fusion_c(c1, x)
                    c1=self.c3(fuse4)
                    x=fuse4
                    soter4 = fuse4
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        img=x
        pyramid = self.get_pyramid(x)


        d40 = self.decoder4(pyramid[3])
        d41= self.attn_block1(d40, pyramid[2])
        d4 = d40 + d41
        d30 = self.decoder3(d4)
        d31 = self.attn_block2(d30, pyramid[1])
        d3 = d30 + d31
        d20 = self.decoder2(d3)
        d21= self.attn_block3(d20, pyramid[0])
        d2 = d20 + d21
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # [2, 2, 224, 320]

        return out
class TD(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        checkpoint = torch.load("D:/PVT-2/PVT-2/pvt_v2_b0.pth")
        #checkpoint = torch.load("D:/FCBFormer-main/FCBFormer-main/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        # filters = [64, 128, 256, 512]
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
        filters = [32, 64, 160, 256]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.attn_block1 = spAttention_block(filters[2], filters[2], filters[2], 4)
        self.attn_block2 = spAttention_block(filters[1], filters[1], filters[1], 4)
        self.attn_block3 = spAttention_block(filters[0], filters[0], filters[0], 4)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

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
        img=x
        pyramid = self.get_pyramid(x)

        d40 = self.decoder4(pyramid[3])
        d41= self.attn_block1(d40, pyramid[2])
        d4 = d40 + d41
        d30 = self.decoder3(d4)
        d31 = self.attn_block2(d30, pyramid[1])
        d3 = d30 + d31
        d20 = self.decoder2(d3)
        d21= self.attn_block3(d20, pyramid[0])
        d2 = d20 + d21
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # [2, 2, 224, 320]

        return out
# def pvt_v2_b1(pretrained=False, **kwargs):
#     model = PyramidVisionTransformerV2(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
class TTG(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = pvt_v2.PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        checkpoint = torch.load("D:/PVT-2/PVT-2/pre/pvt_v2_b1.pth")
        #checkpoint = torch.load("D:/FCBFormer-main/FCBFormer-main/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        # filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4

        # filters = [32, 64, 160, 256]
        filters = [64, 128, 256, 512]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.convd=Conv(320, 256, 1, bn=True, relu=True)

        self.attn_block1 = spAttention_block(filters[2], 320, filters[2], 4)
        self.attn_block2 = spAttention_block(filters[1], filters[1], filters[1], 4)
        self.attn_block3 = spAttention_block(filters[0], filters[0], filters[0], 4)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

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
        img=x
        pyramid = self.get_pyramid(x)

        d40 = self.decoder4(pyramid[3])
        d41= self.attn_block1(d40, pyramid[2])
        d4 = d40 + self.convd(d41)
        d30 = self.decoder3(d4)
        d31 = self.attn_block2(d30, pyramid[1])
        d3 = d30 + d31
        d20 = self.decoder2(d3)
        d21= self.attn_block3(d20, pyramid[0])
        d2 = d20 + d21
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # [2, 2, 224, 320]

        return out
class FCBFormer(nn.Module):
    def __init__(self, size=352):

        super().__init__()

        self.TB = TB()

        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out
from ptflops import get_model_complexity_info
if __name__ == '__main__':

    # model = TTG()
    # input = torch.randn(1, 3, 224, 224)
    # out=model(input)
    #
    net = pdeeplite() #可以为自己搭建的模型
    flops, params = get_model_complexity_info(net, (3,224,224), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)