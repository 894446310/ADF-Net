# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from einops import rearrange
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#
# import numpy as np
# import torch.nn.functional as F
# from utils.cfg_Trans import *
# from utils.MCT import MetaFormer
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         # self.fc1 = nn.Linear(in_features, hidden_features)
#         self.fc1 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
#
#         self.act = act_layer()
#         # self.fc2 = nn.Linear(hidden_features, out_features)
#         self.fc2 = nn.Conv1d(hidden_features, out_features, 3, 1, 1)
#
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
# class LePEAttention(nn.Module):
#     def __init__(self, dim, resolution, idx, split_size=8, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
#                  qk_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out or dim
#         self.resolution = resolution
#         self.split_size = split_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         if idx == -1:
#             H_sp, W_sp = self.resolution, self.resolution
#         elif idx == 0:
#             H_sp, W_sp = self.resolution, self.split_size
#         elif idx == 1:
#             W_sp, H_sp = self.resolution, self.split_size
#         else:
#             print("ERROR MODE", idx)
#             exit(0)
#         self.H_sp = H_sp
#         self.W_sp = W_sp
#         stride = 1
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def im2cswin(self, x):
#         B, N, C = x.shape
#         H = W = int(np.sqrt(N))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = img2windows(x, self.H_sp, self.W_sp)
#         x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
#         return x
#
#     def get_lepe(self, x, func):
#         B, N, C = x.shape
#         H = W = int(np.sqrt(N))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#
#         H_sp, W_sp = self.H_sp, self.W_sp
#         x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
#
#         lepe = func(x)  ### B', C, H', W'
#         lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
#
#         x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
#         return x, lepe
#
#     def forward(self, qkv):
#         """
#         x: B L C
#         """
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         ### Img2Window
#         H = W = self.resolution
#         B, L, C = q.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         q = self.im2cswin(q)
#         k = self.im2cswin(k)
#         v, lepe = self.get_lepe(v, self.get_v)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
#         attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v) + lepe
#         x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
#
#         ### Window2Img
#         x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
#
#         return x
#
#
# class CSWinBlock(nn.Module):
#
#     def __init__(self, dim, reso, num_heads,
#                  split_size=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  last_stage=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.patches_resolution = reso
#         self.split_size = split_size
#         self.mlp_ratio = mlp_ratio
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.norm1 = norm_layer(dim)
#
#         if self.patches_resolution == split_size:
#             last_stage = True
#         if last_stage:
#             self.branch_num = 1
#         else:
#             self.branch_num = 2
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)
#
#         if last_stage:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim, resolution=self.patches_resolution, idx=-1,
#                     split_size=split_size, num_heads=num_heads, dim_out=dim,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         else:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim // 2, resolution=self.patches_resolution, idx=i,
#                     split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
#                        drop=drop)
#         self.norm2 = norm_layer(dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#
#         H = W = self.patches_resolution
#         B, L, C = x.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         img = self.norm1(x)
#         qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
#
#         if self.branch_num == 2:
#             x1 = self.attns[0](qkv[:, :, :, :C // 2])
#             x2 = self.attns[1](qkv[:, :, :, C // 2:])
#             attened_x = torch.cat([x1, x2], dim=2)
#         else:
#             attened_x = self.attns[0](qkv)
#         attened_x = self.proj(attened_x)
#         x = x + self.drop_path(attened_x)
#
#         x = self.norm2(x)
#         x = x.permute(0, 2, 1)
#         x = x + self.drop_path(self.mlp(x))
#         x = x.permute(0, 2, 1)
#
#         return x
#
#
# def img2windows(img, H_sp, W_sp):
#     """
#     img: B C H W
#     """
#     B, C, H, W = img.shape
#     img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#     img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
#     return img_perm
#
#
# def windows2img(img_splits_hw, H_sp, W_sp, H, W):
#     """
#     img_splits_hw: B' H W C
#     """
#     B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
#
#     img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
#     img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return img
#
#
# class Merge_Block(nn.Module):
#     def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
#         self.norm = norm_layer(dim_out)
#
#     def forward(self, x):
#         B, new_HW, C = x.shape
#         H = W = int(np.sqrt(new_HW))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = self.conv(x)
#         B, C = x.shape[:2]
#         x = x.view(B, C, -1).transpose(-2, -1).contiguous()
#         x = self.norm(x)
#
#         return x
#
#
# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#
#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H = W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
#
#         x = x.view(B, H, W, C)
#
#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
#
#         x = self.norm(x)
#         x = self.reduction(x)
#
#         return x
#
#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"
#
#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops
#
#
# class PatchExpand(nn.Module):
#     def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
#         self.norm = norm_layer(dim // dim_scale)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H = W = self.input_resolution
#         x = self.expand(x)
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
#         x = x.view(B, -1, C // 4)
#         x = self.norm(x)
#
#         return x
#
#
# class FinalPatchExpand_X4(nn.Module):
#     def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(dim, 16 * dim, bias=False)
#         self.output_dim = dim
#         self.norm = norm_layer(self.output_dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         x = self.expand(x)
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
#                       c=C // (self.dim_scale ** 2))
#         x = x.view(B, -1, self.output_dim)
#         x = self.norm(x)
#
#         return x
#
#
# class BasicLayer(nn.Module):
#     """ A basic CSwin Transformer layer for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             CSWinBlock(dim, input_resolution,
#                        num_heads=num_heads,
#                        mlp_ratio=mlp_ratio,
#                        qkv_bias=qkv_bias, qk_scale=qk_scale,
#                        drop=drop, attn_drop=attn_drop,
#                        # drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                        drop_path=drop_path[i],
#                        norm_layer=norm_layer,
#                        )
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#
#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops
#
#
# class BasicLayer_up(nn.Module):
#     """ A basic CSwin Transformer layer for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             CSWinBlock(dim, input_resolution,
#                        num_heads=num_heads,
#                        mlp_ratio=mlp_ratio,
#                        qkv_bias=qkv_bias, qk_scale=qk_scale,
#                        drop=drop, attn_drop=attn_drop,
#                        # drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                        drop_path=drop_path[i],
#                        norm_layer=norm_layer,
#                        )
#             for i in range(depth)])
#
#         # patch merging layer
#         if upsample is not None:
#             self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
#         else:
#             self.upsample = None
#
#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.upsample is not None:
#             x = self.upsample(x)
#         return x
#
#
# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding
#
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """
#
#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]
#
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
#         return x
#
#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops
#
#
# class CSwinTransformerSys(nn.Module):
#     r""" Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030
#
#     Args:
#         img_size (int | tuple(int)): Input image size. Default 224
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#     """
#
#     def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2, height=128,
#                  embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, final_upsample="expand_first", **kwargs):
#         super().__init__()
#
#
#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.num_features_up = int(embed_dim * 2)
#         self.mlp_ratio = mlp_ratio
#         self.final_upsample = final_upsample
#
# # ----------------------------FIT------------------------------------
# #         self.FIT = MetaFormer(num_skip=4, skip_dim=[96, 192, 384, 768])
#         self.FIT = MetaFormer(num_skip=3, skip_dim=[192, 384, 768])
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#
#         # build encoder and bottleneck layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#                                input_resolution=(patches_resolution[0] // (2 ** i_layer)),
#                                depth=depths[i_layer],
#                                num_heads=num_heads[i_layer],
#                                window_size=window_size,
#                                mlp_ratio=self.mlp_ratio,
#                                qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                norm_layer=norm_layer,
#                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                use_checkpoint=use_checkpoint)
#             self.layers.append(layer)
#
#         # build decoder layers
#         self.layers_up = nn.ModuleList()
#         self.concat_back_dim = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
#                                       int(embed_dim * 2 ** (
#                                               self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
#             if i_layer == 0:
#                 layer_up = PatchExpand(
#                     input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer))),
#                     dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
#             else:
#                 layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
#                                          input_resolution=(
#                                                  patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer))),
#                                          depth=depths[(self.num_layers - 1 - i_layer)],
#                                          num_heads=num_heads[(self.num_layers - 1 - i_layer)],
#                                          window_size=window_size,
#                                          mlp_ratio=self.mlp_ratio,
#                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                          drop=drop_rate, attn_drop=attn_drop_rate,
#                                          drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
#                                              depths[:(self.num_layers - 1 - i_layer) + 1])],
#                                          norm_layer=norm_layer,
#                                          upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
#                                          use_checkpoint=use_checkpoint)
#             self.layers_up.append(layer_up)
#             self.concat_back_dim.append(concat_linear)
#
#         self.norm = norm_layer(self.num_features)
#         self.norm_up = norm_layer(self.embed_dim)
#
#         if self.final_upsample == "expand_first":
#             print("---final upsample expand_first---")
#             self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
#                                           dim_scale=4, dim=embed_dim)
#             self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
#
#         self.apply(self._init_weights)
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
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     # Encoder and Bottleneck
#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#         x_downsample = []
#
#         i = 0
#         for layer in self.layers:
#             x_downsample.append(x)
#             x = layer(x)
#
#         x = self.norm(x)  # B L C
#
#         return x, x_downsample
#
#     # Decoder and Skip connection
#     def forward_up_features(self, x, x_downsample):
#         for inx, layer_up in enumerate(self.layers_up):
#             if inx == 0:
#                 x = layer_up(x)
#             else:
#                 x = torch.cat([x, x_downsample[3 - inx]], -1)
#                 x = self.concat_back_dim[inx](x)
#                 x = layer_up(x)
#
#         x = self.norm_up(x)  # B L C
#
#         return x
#
#     def up_x4(self, x):
#         H, W = self.patches_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input features has wrong size"
#
#         if self.final_upsample == "expand_first":
#             x = self.up(x)
#             x = x.view(B, 4 * H, 4 * W, -1)
#             x = x.permute(0, 3, 1, 2)  # B,C,H,W
#             x = self.output(x)
#
#         return x
#
#     def forward(self, x):
#         x, x_downsample = self.forward_features(x)
#         x1,x2,x3,x4 = x_downsample[0],x_downsample[1],x_downsample[2],x_downsample[3]
#
#         B,L1,C1 = x1.shape
#         x1 = x1.permute(0,2,1).view(B,C1,int(L1**0.5),int(L1**0.5))
#
#         B,L2,C2 = x2.shape
#         x2 = x2.permute(0,2,1).view(B,C2,int(L2**0.5),int(L2**0.5))
#
#         B,L3,C3 = x3.shape
#         x3 = x3.permute(0,2,1).view(B,C3,int(L3**0.5),int(L3**0.5))
#
#         B,L4,C4 = x4.shape
#         x4 = x4.permute(0,2,1).view(B,C4,int(L4**0.5),int(L4**0.5))
#
#         x1,x2,x3,x4 = self.FIT(x1,x2,x3,x4)
#
#         x1 = x1.permute(0,2,3,1).view(B,L1,C1)
#         x2 = x2.permute(0,2,3,1).view(B,L2,C2)
#         x3 = x3.permute(0,2,3,1).view(B,L3,C3)
#         x4 = x4.permute(0,2,3,1).view(B,L4,C4)
#
#         x_downsample_list = [x1,x2,x3,x4]
#
#         x = self.forward_up_features(x, x_downsample_list)
#         x = self.up_x4(x)
#
#         return x
#
#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops
#
#
# class CSwin_Unet(nn.Module):
#     def __init__(self, config, num_classes=2, zero_head=False, vis=False):
#         super(CSwin_Unet, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.config = config
#
#         self.CSwin_Unet = CSwinTransformerSys(img_size=config.IMG_SIZE,
#                                              patch_size=config.PATCH_SIZE,
#                                              in_chans=config.IN_CHANS,
#                                              num_classes=self.num_classes,
#                                              embed_dim=config.EMBED_DIM,
#                                              depths=config.DEPTHS,
#                                              depths_decoder=config.depths_decoder,
#                                              num_heads=config.NUM_HEADS,
#                                              window_size=config.WINDOW_SIZE,
#                                              mlp_ratio=config.MLP_RATIO,
#                                              qkv_bias=config.QKV_BIAS,
#                                              qk_scale=config.QK_SCALE,
#                                              drop_rate=config.DROP_RATE,
#                                              drop_path_rate=config.DROP_PATH_RATE,
#                                              ape=config.APE,
#                                              patch_norm=config.PATCH_NORM,
#                                              use_checkpoint=config.USE_CHECKPOINT)
#
#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         logits = self.CSwin_Unet(x)
#         return logits
#
# import torch
# import utils.cfg_Trans as cfg_Trans
# SLTnet =CSwin_Unet(cfg_Trans, num_classes=1)
#
#
# from ptflops import get_model_complexity_info
# if __name__ == '__main__':
#         import torch
#         import utils.cfg_Trans as cfg_Trans
#
#         print('-----' * 5)
#         rgb = torch.randn(2, 3, 512, 512)
#         cs = CSwin_Unet(cfg_Trans, num_classes=1)
#         out = cs(rgb)
#
#         print(out.shape)
#
#         net =CSwin_Unet(cfg_Trans, num_classes=1) # 可以为自己搭建的模型
#         flops, params = get_model_complexity_info(net, (3, 512, 512), as_strings=True,
#                                                   print_per_layer_stat=True)  # (3,512,512)输入图片的尺寸
#         print("Flops: {}".format(flops))
#         print("Params: " + params)
