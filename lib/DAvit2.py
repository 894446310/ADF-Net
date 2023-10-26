import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from functools import partial
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

        filters = [96, 192, 384, 768]
        # resnet = models.resnet34(pretrained=True)
        #
        # self.firstconv=resnet.conv1
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

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        #
        #
        #
        #
        # self.fusion_c = BiFusion_block(ch_1=512, ch_2=768, r_2=4, ch_int=512, ch_out=512)
        # self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256)
        # self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=192, r_2=4, ch_int=128, ch_out=128)
        # self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=96, r_2=4, ch_int=64, ch_out=64)
        #
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)


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


        d4 = self.decoder4(x_t) + x_t_1  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d3 = self.decoder3(d4) + x_t_2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d2 = self.decoder2(d3) + x_t_3  # [2, 64, 56, 80]
        d1 = self.decoder1(d2)  # [2, 64, 112, 160]



        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        return out

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DaViT, self).train(mode)
