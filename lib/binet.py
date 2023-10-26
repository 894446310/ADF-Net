"""
BiFormer impl.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

from ops.bra_legacy import BiLevelRoutingAttention

from models._common import Attention, AttentionLePE, DWConv

from torchvision.models import resnet34 as resnet
from torchvision import models

# from positional_encodings import PositionalEncodingPermute2D, Summer
# from siren_pytorch import SirenNet
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


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


class AFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int,
                 drop_rate=0.1):
        super(AFF_block, self).__init__()

        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (256到256）
        self.W_x = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (384，256)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = ch_int
        r = 8
        L = 48
        d = max(int(self.inplanes / r), L)
        self.W1 = nn.Linear(2 * self.inplanes, d)
        self.W2 = nn.Linear(d, self.inplanes)
        self.W3 = nn.Linear(d, self.inplanes)  # x=，g=t
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x, g):
        U1 = self.W_g(g)
        U2 = self.W_x(x)
        U = U1 + U2
        avg_pool = F.avg_pool2d(U1, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        max_pool = F.max_pool2d(U2, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
        zc1 = self.W2(nn.ReLU()(self.W1(sc)))
        zc2 = self.W3(nn.ReLU()(self.W1(sc)))
        ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        fuse = U1 * ac1 + U2 * ac2+U
        #fuse=self.dropout(fuse)
        return fuse


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
        a = x
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


nonlinearity = partial(F.relu, inplace=True)


class spAttention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, r_2):
        super(spAttention_block, self).__init__()
        # self.W_g = nn.Sequential(
        #     nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(F_int)
        # )
        # self.W_x = nn.Sequential(
        #     nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(F_int)
        # )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.GELU()

    def forward(self, g, x):
        f = g+x
        psi = self.relu(f)
        psi = self.psi(psi)
        return x * psi

      # self.attn_block1 = spAttention_block(256, 256, 256, 4)
      #   self.attn_block2 = spAttention_block(128, 128, 128, 4)
      #   self.attn_block3 = spAttention_block(64, 64, 64, 4)
      #   self.decoder4 = DecoderBlock(512, filters[2])
      #   self.decoder3 = DecoderBlock(filters[2], filters[1])
      #   self.decoder2 = DecoderBlock(filters[1], filters[0])
      #   self.decoder1 = DecoderBlock(filters[0], filters[0])
    #self.FAD=FADBlock(512, filters[2],256, 256, 256, 4)
    # d40 = self.decoder4(x_c)
    # d41 = self.attn_block1(d40, x_c_1)
    # d4 = d40 + d41
    #
    # d30 = self.decoder3(d4)
    # d31 = self.attn_block2(d30, x_c_2)
    # d3 = d30 + d31
    #
    # d20 = self.decoder2(d3)
    # d21 = self.attn_block3(d20, x_c_3)
    # d2 = d20 + d21
    # d4 = self.FAD(x_c, x_c_1)
    # d3 = self.FAD2(d4, x_c_2)
    # d2 = self.FAD3(d3, x_c_2)

class FAD_block(nn.Module):
    def __init__(self, in_channels, n_filters, F_g, F_l, F_int, r_2):
        super(FAD_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.GELU()

    def forward(self, x, g):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        f = g+x
        psi = self.relu(f)
        psi = self.psi(psi)
        return g * psi


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class BiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)
            outs.append(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x, outs

    def forward(self, x):
        x, outs = self.forward_features(x)
        return x, outs


#################### model variants #######################


model_urls = {
    "biformer_tiny_in1k": "https://matix.li/e36fe9fb086c",
    "biformer_small_in1k": "https://matix.li/5bb436318902",
    "biformer_base_in1k": "https://matix.li/995db75f585d",
}


# https://github.com/huggingface/pytorch-image-models/blob/4b8cfa6c0a355a9b3cb2a77298b240213fb3b921/timm/models/_factory.py#L93

@register_model
def biformer_tiny(pretrained=True, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model_key = 'biformer_tiny_in1k'
        url = model_urls[model_key]
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
        #checkpoint = torch.load("D:/BiFormer-public_release/BiFormer-public_release/biformer_tiny_best.pth")
        checkpoint = torch.load("pretrained/biformer_tiny_best.pth")
        model.load_state_dict(checkpoint["model"])
    return model


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
class ADF_Net(nn.Module):
    def __init__(self, img_ch=3, normal_init=True, output_ch=1):
        super(ADF_Net, self).__init__()

        self.bone = biformer_tiny(pretrained=True)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        filters = [64, 128, 256, 512]

        self.AFF = AFF_block(ch_1=512, ch_2=512, ch_int=512)
        self.AFF1 = AFF_block(ch_1=256, ch_2=256, ch_int=256)
        self.AFF2 = AFF_block(ch_1=128, ch_2=128, ch_int=128)
        self.AFF3 = AFF_block(ch_1=64, ch_2=64, ch_int=64)


        self.FAD = FAD_block(512, filters[2], filters[2], filters[2], filters[2], 4)
        self.FAD1 = FAD_block(filters[2], filters[1], filters[1], filters[1], filters[1], 4)
        self.FAD2 = FAD_block(filters[1], filters[0], filters[0], filters[0], filters[0], 4)
        self.decoder1 = DecoderBlock(filters[0], filters[0])
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
        # if normal_init:
        #     self.init_weights()

    def forward(self, x):
        # encoding path
        c1 = x
        t3, outs = self.bone(x)
        outs[3] = t3

        c1 = self.firstconv(c1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        c1 = self.firstbn(c1)
        c1 = self.firstrelu(c1)
        c1 = self.firstmaxpool(c1)  # [2, 64, 56, 80]
        x_u_3 = self.encoder1(c1)  # [2, 64, 56, 80]
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]


        x_c = self.AFF(x_u, outs[3])  # 512,1/32
        x_c_1 = self.AFF1(x_u_1, outs[2])  # 256
        x_c_2 = self.AFF2(x_u_2, outs[1])  # 128
        x_c_3 = self.AFF3(x_u_3, outs[0])  # 64
        d4=self.FAD(x_c, x_c_1)
        d3=self.FAD1(d4,x_c_2)
        d2 = self.FAD2(d3,x_c_3)

        d1 = self.decoder1(d2)  # [2, 64, 112, 160]
        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]


        map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(d2), scale_factor=4, mode='bilinear', align_corners=True)
        return map_x, map_1, out





def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# @register_model
# def biformer_small(pretrained=False, pretrained_cfg=None,
#                    pretrained_cfg_overlay=None, **kwargs):
#     model = BiFormer(
#         depth=[4, 4, 18, 4],
#         embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
#         #------------------------------
#         n_win=7,
#         kv_downsample_mode='identity',
#         kv_per_wins=[-1, -1, -1, -1],
#         topks=[1, 4, 16, -2],
#         side_dwconv=5,
#         before_attn_dwconv=3,
#         layer_scale_init_value=-1,
#         qk_dims=[64, 128, 256, 512],
#         head_dim=32,
#         param_routing=False, diff_routing=False, soft_routing=False,
#         pre_norm=True,
#         pe=None,
#         #-------------------------------
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     if pretrained:
#         model_key = 'biformer_small_in1k'
#         url = model_urls[model_key]
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
#         model.load_state_dict(checkpoint["model"])
#
#     return model
#
#
# @register_model
# def biformer_base(pretrained=False, pretrained_cfg=None,
#                   pretrained_cfg_overlay=None, **kwargs):
#     model = BiFormer(
#         depth=[4, 4, 18, 4],
#         embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
#         # use_checkpoint_stages=[0, 1, 2, 3],
#         use_checkpoint_stages=[],
#         #------------------------------
#         n_win=7,
#         kv_downsample_mode='identity',
#         kv_per_wins=[-1, -1, -1, -1],
#         topks=[1, 4, 16, -2],
#         side_dwconv=5,
#         before_attn_dwconv=3,
#         layer_scale_init_value=-1,
#         qk_dims=[96, 192, 384, 768],
#         head_dim=32,
#         param_routing=False, diff_routing=False, soft_routing=False,
#         pre_norm=True,
#         pe=None,
#         #-------------------------------
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     if pretrained:
#         model_key = 'biformer_base_in1k'
#         url = model_urls[model_key]
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
#         model.load_state_dict(checkpoint["model"])
#
#     return model

# if __name__ == '__main__':
#     model = Bi_Net()
#     input = torch.randn(1, 3, 224, 224)
#     out = model(input)
#     print(out.size())

# from thop import profile
#
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(Bi_Net() , inputs=(input,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

from ptflops import get_model_complexity_info

if __name__ == '__main__':
    net =ADF_Net()  # 可以为自己搭建的模型
    flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                              print_per_layer_stat=True)  # (3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)