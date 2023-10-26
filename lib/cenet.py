import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
# from ..builder import BACKBONES
from torchvision import models
from functools import partial

# from ptflops import get_model_complexity_info
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
class BiFusion_block(nn.Module):   #BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512)
    def  __init__(self, ch_1, ch_2, r_2, ch_int, ch_out,
                 drop_rate=0.):  # (ch_1=256, ch_2=384（通道） r_2=4（通道）, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        super(BiFusion_block, self).__init__()
        # coor(ch2,r2)(384,4)trans
        # self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # (256到256）
        # self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # (384，256)
        self.W_g = Conv(512, 512, 1, bn=True, relu=False)  # (256到256）
        self.W_x = Conv(512, 512, 1, bn=True, relu=False)  # (384，256)
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

        # U1 = self.W_g(g)
        # U2 = self.W_x(x)

        U1 = g
        U2 = x

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
        fuse = U1 * ac1 + U2 * ac2+U2

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
            if self.rasa_cfg.atrous_rates is not None:
                self.ds = ds_conv2d(
                    dim, dim, kernel_size=3, stride=1,
                    dilation=self.rasa_cfg.atrous_rates, groups=dim, bias=qkv_bias,
                    act_layer=self.rasa_cfg.act_layer, init=self.rasa_cfg.init,
                )
            if self.rasa_cfg.r_num > 1:
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
            if self.rasa_cfg.atrous_rates is not None:
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
            x = self._inner_attention(x)
            if self.rasa_cfg.r_num > 1:
                x = self.silu(x)
            for _ in range(self.rasa_cfg.r_num - 1):
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
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        else:
            x = x.permute(0, 3, 1, 2)
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        return x


class lite_vision_transformer(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], in_chans=3, num_classes=1, classes=1, channels=3,patch_size=4,
                 embed_dims=[64, 64, 160, 256], num_heads=[2, 2, 5, 8],
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=None,
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

        filters = [64, 64, 160, 256]
        # resnet = models.resnet34(pretrained=True)
        # # conresnet=OD_ResNet(BasicBlock, [3, 4, 6, 3])
        # # conresnet=convres
        # self.firstconv = resnet.conv1
        # # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        # # self.firstmaxpool = resnet.maxpool
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4

        self.firstconv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # downsampling
        self.conv = conv_block(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = conv_block(64, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(64, 160)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(160, 256, drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.fusion_c = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=512, ch_out=512)
        self.fusion_c1 = BiFusion_block(ch_1=160, ch_2=160, r_2=4, ch_int=256, ch_out=256)
        self.fusion_c2 = BiFusion_block(ch_1=64, ch_2=64, r_2=4, ch_int=128, ch_out=128)
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
        d1=x
        outs = []
        c=x
        for idx, stage in enumerate(self.backbone):  #
            x = stage(x)
            outs.append(x.permute(0, 3, 1, 2).contiguous())

        x = self.norm(x)
        outs[3]=x
        d1 = self.firstconv(d1)  # [2, 3, 224, 320] --> [2, 64, 112, 160]
        d1 = self.firstbn(d1)
        d1 = self.firstrelu(d1)
        d1 = self.firstmaxpool(d1)  # [2, 64, 56, 80]
        d1 = self.conv(d1)  # [2, 64, 56, 80]   1/4
        fuse1 = self.fusion_c3(d1, outs[0])

        maxpool1 = self.maxpool1(fuse1)
        c1 = self.conv1(maxpool1)  # [2, 128, 28, 40]   1/8
        fuse2 = self.fusion_c2(c1, outs[1])

        maxpool2 = self.maxpool2(fuse2)
        c1 = self.conv2(maxpool2)  # [2, 256, 28, 40]  1/16
        fuse3 = self.fusion_c1(c1, outs[2])

        maxpool3 = self.maxpool3(fuse3)
        c1 = self.conv3(maxpool3)
        fuse4 = self.fusion_c(c1, outs[3])

        d4 = self.decoder4(fuse4) + fuse3  # [2, 256, 14, 20] + [2, 256, 14, 20] --> [2, 256, 14, 20]
        d3 = self.decoder3(d4) + fuse2  # [2, 128, 28, 40] + [2, 128, 28, 40]
        d2 = self.decoder2(d3) + fuse1  # [2, 64, 56, 80]
        d1 = self.decoder1(d2)  # [2, 64, 112, 160]

        out = self.finaldeconv1(d1)  # [2, 32, 224, 320]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2, 32, 224, 320]
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # [2, 2, 224, 320]
        return out



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
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
# from ptflops import get_model_complexity_info


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class GatedAttentionUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2) // 2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size),
            nn.GELU()
        )

        self.cw = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        x1, x2 = self.w1(x), self.w2(x)
        out = self.wo(x1 * x2) + self.cw(x)
        return out


class DilatedGatedAttention(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=[7, 5, 2, 1]):
        super().__init__()

        self.mda0 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[0] - 1)) // 2,
                              dilation=dilated_ratio[0], groups=in_c // 4)
        self.mda1 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[1] - 1)) // 2,
                              dilation=dilated_ratio[1], groups=in_c // 4)
        self.mda2 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[2] - 1)) // 2,
                              dilation=dilated_ratio[2], groups=in_c // 4)
        self.mda3 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[3] - 1)) // 2,
                              dilation=dilated_ratio[3], groups=in_c // 4)
        self.norm_layer = nn.GroupNorm(4, in_c)
        self.conv = nn.Conv2d(in_c, in_c, 1)

        self.gau = GatedAttentionUnit(in_c, out_c, 3)

    def forward(self, x):
        x = torch.chunk(x, 4, dim=1)
        x0 = self.mda0(x[0])
        x1 = self.mda1(x[1])
        x2 = self.mda2(x[2])
        x3 = self.mda3(x[3])
        x = F.gelu(self.conv(self.norm_layer(torch.cat((x0, x1, x2, x3), dim=1))))
        x = self.gau(x)
        return x


class EAblock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x)
        return x


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class MALUNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            EAblock(c_list[2]),
            DilatedGatedAttention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            EAblock(c_list[3]),
            DilatedGatedAttention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            EAblock(c_list[4]),
            DilatedGatedAttention(c_list[4], c_list[5]),
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            DilatedGatedAttention(c_list[5], c_list[4]),
            EAblock(c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            DilatedGatedAttention(c_list[4], c_list[3]),
            EAblock(c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            DilatedGatedAttention(c_list[3], c_list[2]),
            EAblock(c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return out0


if __name__ == '__main__':
    # net = MALUNet() #可以为自己搭建的模型
    # flops, params = get_model_complexity_info(net, (3,224,320), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    # print("Flops: {}".format(flops))
    # print("Params: " + params)

    from thop import profile
    import torch
    import torchvision

    print('==> Building model..')
    model = MALUNet()
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, (input,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

if __name__ == '__main__':
    net = lite_vision_transformer() #可以为自己搭建的模型
    flops, params = get_model_complexity_info(net, (3,224,320), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)

    # from thop import profile
    # import torch
    # import torchvision
    #
    # print('==> Building model..')
    # model = lite_vision_transformer()
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, (input,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
    # input = torch.rand(2, 3, 224, 320)
    # # model = RCEM(3, 64)
    # # model = BasicBlock(3, 64, 64)
    # model = lite_vision_transformer()
    # out12 = model(input)
    # print(out12)