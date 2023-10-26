import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from torchvision.models import resnet34 as resnet
from torchvision import models
from lib import pvt_v2

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
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
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
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
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
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

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

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

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
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
#         self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=True)  # (256到256）
#         self.W_x = Conv(ch_1, ch_int, 1, bn=True, relu=True)  # (384，256)
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
#
#         # fuse = g + x
#         U = U1 + U2
#
#         avg_pool = F.avg_pool2d(U1, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
#         max_pool = F.max_pool2d(U2, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
#
#         sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
#
#         zc1 = self.W2(nn.ReLU()(self.W1(sc)))
#         zc2 = self.W3(nn.ReLU()(self.W1(sc)))
#
#         ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
#         ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
#         fuse = U1 * ac1 + U2 * ac2 + U
#
#         return fuse
class BiFusion_block(nn.Module):  # BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()

        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (256到256）
        self.W_x = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (384，256)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

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
        self.W3 = nn.Linear(d, self.inplanes)  #x=，g=t

    def forward(self, x, g):
        U1 = self.W_g(g)
        U2 = self.W_x(x)
        U3=self.W(U1*U2)
        #U = U1 + U2 + U3
        U = self.residual(torch.cat([g, x, U3], 1))
        avg_pool = F.avg_pool2d(U1, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        max_pool = F.max_pool2d(U2, (U.size(2), U.size(3)), stride=(U.size(2), U.size(3)))
        sc = torch.sigmoid(torch.cat([avg_pool, max_pool], 1)).reshape([-1, self.inplanes * 2])
        zc1 = self.W2(nn.ReLU()(self.W1(sc)))
        zc2 = self.W3(nn.ReLU()(self.W1(sc)))
        ac1 = (torch.exp(zc1) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        ac2 = (torch.exp(zc2) / (torch.exp(zc1) + torch.exp(zc2))).reshape([-1, self.inplanes, 1, 1])
        fuse = U1 * ac1 + U2 * ac2 + U
        return U2

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
        a=x
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
        f = g + x
        psi = self.relu(f)
        psi = self.psi(psi)
        return x * psi


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
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
            out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x, outs

    def forward(self, x):
        x, outs = self.forward_features(x)
        # x = self.head(x)

        return x, outs


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


# @register_model
def pvt_v2_b0(pretrained=True, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
    # model.default_cfg = _cfg()

    checkpoint = torch.load("D:/PVT-2/PVT-2/pvt_v2_b0.pth")
    model.default_cfg = _cfg()
    model.load_state_dict(checkpoint)

    return model


class P_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(P_Net, self).__init__()

        self.bone = pvt_v2_b0()
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=256, ch_2=160, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=128, ch_2=64, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=64, ch_2=32, r_2=4, ch_int=64, ch_out=64)
        filters = [64, 128, 256, 512]
        self.attn_block1 = spAttention_block(256, 256, 256, 4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

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
        x_u_2 = self.encoder2(x_u_3)  # [2, 128, 28, 40]
        x_u_1 = self.encoder3(x_u_2)  # [2, 256, 14, 20]
        x_u = self.encoder4(x_u_1)  # [2, 512, 7, 10]
        x_c = self.fusion_c(x_u, outs[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x_u_1, outs[2])  # 256
        x_c_2 = self.fusion_c2(x_u_2, outs[1])  # 128
        x_c_3 = self.fusion_c3(x_u_3, outs[0])  # 64

        d40 = self.decoder4(x_c)
        d41, p1, ff1 = self.attn_block1(d40, x_c_1)
        d4 = d40 + d41

        d30 = self.decoder3(d4)
        d31, p2, ff2 = self.attn_block2(d30, x_c_2)
        d3 = d30 + d31

        d20 = self.decoder2(d3)
        d21, p3, ff3 = self.attn_block3(d20, x_c_3)
        d2 = d20 + d21

        d1 = self.decoder1(d2)  # [2, 64, 112, 160]
        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        return out

class P_Netv2(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(P_Netv2, self).__init__()

        self.bone = pvt_v2_b0()
        self.pfc = PFC(64)
        self.img_channels = 3
        self.n_classes = 1
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4



        self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=512, ch_2=160, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=512, ch_2=64, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=256, ch_2=32, r_2=4, ch_int=64, ch_out=64)
        filters = [64, 128, 256, 512]
        self.attn_block1 = spAttention_block(256, 256, 256, 4)
        self.attn_block2 = spAttention_block(128, 128, 128, 4)
        self.attn_block3 = spAttention_block(64, 64, 64, 4)
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # encoding path
        c1 = x
        t3, outs = self.bone(x)
        outs[3] = t3

        x1 = self.pfc(x)
        x2 = self.maxpool(x1)

        x3 = self.down1(x2)
        x4 = self.maxpool(x3)

        x5 = self.down2(x4)
        x6 = self.maxpool(x5)

        x7 = self.down3(x6)
        x8 = self.maxpool(x7)

        x9 = self.down4(x8)
        x10 = self.maxpool(x9)


        x_c = self.fusion_c(x10, outs[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x9, outs[2])  # 256
        x_c_2 = self.fusion_c2(x7, outs[1])  # 128
        x_c_3 = self.fusion_c3(x5, outs[0])  # 64

        d40 = self.decoder4(x_c)
        d41, p1, ff1 = self.attn_block1(d40, x_c_1)
        d4 = d40 + d41

        d30 = self.decoder3(d4)
        d31, p2, ff2 = self.attn_block2(d30, x_c_2)
        d3 = d30 + d31

        d20 = self.decoder2(d3)
        d21, p3, ff3 = self.attn_block3(d20, x_c_3)
        d2 = d20 + d21

        d1 = self.decoder1(d2)  # [2, 64, 112, 160]
        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        return out

class TF(nn.Module):
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

        self.pfc = PFC(64)
        self.img_channels = 3
        self.n_classes = 1
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4


        # filters = [32, 64, 160, 256]
        filters = [64, 128, 256, 512]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=512, ch_2=160, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=512, ch_2=64, r_2=4, ch_int=128, ch_out=128)
        self.fusion_c3 = BiFusion_block(ch_1=256, ch_2=32, r_2=4, ch_int=64, ch_out=64)

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

        x1 = self.pfc(img)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        x9 = self.down4(x8)
        x10 = self.maxpool(x9)  # [2, 512, 7, 10]

        # Decoder
        x_c = self.fusion_c(x10, pyramid[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x9, pyramid[2])  # 256
        x_c_2 = self.fusion_c2(x7, pyramid[1])  # 128
        x_c_3 = self.fusion_c3(x5, pyramid[0])  # 64

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

class TG(nn.Module):
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

        self.pfc = PFC(64)
        self.img_channels = 3
        self.n_classes = 1
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4


        filters = [32, 64, 160, 256]
        # filters = [64, 128, 256, 512]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=512, ch_2=256, r_2=4, ch_int=filters[3], ch_out=filters[3])
        self.fusion_c1 = BiFusion_block(ch_1=512, ch_2=160, r_2=4, ch_int=filters[2], ch_out=filters[2])
        self.fusion_c2 = BiFusion_block(ch_1=512, ch_2=64, r_2=4, ch_int=filters[1], ch_out=filters[1])
        self.fusion_c3 = BiFusion_block(ch_1=256, ch_2=32, r_2=4, ch_int=filters[0], ch_out=filters[0])

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

        x1 = self.pfc(img)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        x9 = self.down4(x8)
        x10 = self.maxpool(x9)  # [2, 512, 7, 10]

        # Decoder
        x_c = self.fusion_c(x10, pyramid[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x9, pyramid[2])  # 256
        x_c_2 = self.fusion_c2(x7, pyramid[1])  # 128
        x_c_3 = self.fusion_c3(x5, pyramid[0])  # 64

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



class TT(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        checkpoint = torch.load("D:/PVT-2/PVT-2/pvt_v2_b0.pth")
        # checkpoint = torch.load("D:/FCBFormer-main/FCBFormer-main/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
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




        filters = [32, 64, 160, 256]
        # filters = [64, 128, 256, 512]
        drop_rate = 0.2
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=320, ch_2=256, r_2=4, ch_int=filters[3], ch_out=filters[3])
        self.fusion_c1 = BiFusion_block(ch_1=112, ch_2=160, r_2=4, ch_int=filters[2], ch_out=filters[2])
        self.fusion_c2 = BiFusion_block(ch_1=40, ch_2=64, r_2=4, ch_int=filters[1], ch_out=filters[1])
        self.fusion_c3 = BiFusion_block(ch_1=24, ch_2=32, r_2=4, ch_int=filters[0], ch_out=filters[0])

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
        img = x
        pyramid = self.get_pyramid(x)
        x1 = self.start_conv(img)
        x2 = self.down_block_2(x1)
        x3 = self.down_block_3(x2)
        x4 = self.down_block_4(x3)
        x5 = self.down_block_5(x4)
        # Decoder
        x_c = self.fusion_c(x5, pyramid[3])  # 512,1/32
        x_c_1 = self.fusion_c1(x4, pyramid[2])  # 256
        x_c_2 = self.fusion_c2(x3, pyramid[1])  # 128
        x_c_3 = self.fusion_c3(x2, pyramid[0])  # 64

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

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]

        return out

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




@register_model
def pvt_v2_b1(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b3(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b4(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b5(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


# @register_model
def pvt_v2_b2_li(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()

    return model
import torch.nn as nn
import torch.nn.functional as F
import torch
from ca.encoder import CSA

csa_block = CSA()


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
        self.down5 = csa_block.layer5
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

from ptflops import get_model_complexity_info

if __name__ == '__main__':
    # model = pvt_v2_b0()
    # model = TT()
    # input = torch.randn(1, 3, 224, 224)
    # out = model(input)
    # print(out.size())
    #
    net = TT() #可以为自己搭建的模型
    flops, params = get_model_complexity_info(net, (3,224,224), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)