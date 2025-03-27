import os
import warnings
from functools import partial
import numpy as np
import math

import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_

from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention


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
        x = self.dwconv(x, H, W)
        x = self.act(x)
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
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class FHARD(nn.Module):
    """
    CASA module
    """
    def __init__(self, in_channel):
        super(FHARD, self).__init__()
        self.rgb_branch1_ca = ChannelAttention(np.int(in_channel / 2))
        self.r_branch1_ca = ChannelAttention(np.int(in_channel / 2))
        self.rgb_branch1_sa = SpatialAttention()
        self.r_branch1_sa = SpatialAttention()

        self.conv5 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x1_rgb, x1_d):


        x1_rgb_ca_f = self.rgb_branch1_ca(x1_rgb)
        x1_d_ca_f = self.r_branch1_ca(x1_d)
        x1_rgbd_ca_f = x1_d_ca_f.mul(x1_rgb_ca_f)
        x1_rgbd_ca_fr = x1_rgbd_ca_f + x1_rgb_ca_f
        x1_rgbd_ca_fd = x1_rgbd_ca_f + x1_d_ca_f

        x1_rgb_ca = x1_rgb.mul(x1_rgbd_ca_fr)
        x1_d_ca = x1_d.mul(x1_rgbd_ca_fd)

        x1_rgb_sa_f = self.rgb_branch1_sa(x1_rgb_ca)
        x1_d_sa_f = self.r_branch1_sa(x1_d_ca)
        x1_rgbd_sa_f = x1_rgb_sa_f.mul(x1_d_sa_f)
        x1_rgbd_sa_fr = torch.cat((x1_rgbd_sa_f, x1_rgb_sa_f), 1)
        x1_rgbd_sa_fd = torch.cat((x1_rgbd_sa_f, x1_d_sa_f), 1)
        x1_rgbd_sa_fr = self.conv5(x1_rgbd_sa_fr)
        x1_rgbd_sa_fd = self.conv5(x1_rgbd_sa_fd)

        return x1_rgbd_sa_fr, x1_rgbd_sa_fd




class HAIM(nn.Module):
    """
    CCE module
    """
    def __init__(self, in_channel):
        super(HAIM, self).__init__()

        self.relu = nn.ReLU(True)
        self.ca = ChannelAttention(in_channel)
        self.rdf = FHARD(np.int(in_channel))
        self.ca1 = ChannelAttention(np.int(in_channel))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, np.int(in_channel/2), kernel_size=3, padding=1),
            nn.BatchNorm2d(np.int(in_channel/2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_rgb, x_d):

        x1_rgb = self.conv1(x_rgb)

        x1_d = self.conv1(x_d)

        x1_rgbd_sa_fr, x1_rgbd_sa_fd = self.rdf(x1_rgb, x1_d)
        x1_d_sa = x1_d.mul(x1_rgbd_sa_fr)
        x1_d1 = x1_d + x1_d_sa
        x1_rgbd_sa_fr1, x1_rgbd_sa_fd1 = self.rdf(x1_rgb, x1_d1)
        x1_rgb_sa = x1_rgb.mul(x1_rgbd_sa_fd1)

        x1_rgbd_sa_fr2, x1_rgbd_sa_fd2 = self.rdf(x1_rgb_sa, x1_d)
        x1_d_sa1 = x1_d.mul(x1_rgbd_sa_fr2)
        x1_d2 = x1_d + x1_d_sa1
        x1_rgbd_sa_fr3, x1_rgbd_sa_fd3 = self.rdf(x1_rgb, x1_d2)
        x1_rgb_sa1 = x1_rgb.mul(x1_rgbd_sa_fd3)

        y = torch.cat((x1_rgb_sa1,  x1_rgb,), 1)
        y_ca = y.mul(self.ca(y))

        z = y_ca + x_rgb

        return z



class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        #(B, C, H, W)->(B, C, H * W)->(B, H * W, C)
        x = self.norm(x)

        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,incan=[64,128,320,512],
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        self.time_embed = nn.ModuleList()
        for i in range(0, len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.patch_embed11 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                                              embed_dim=embed_dims[0])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])


        self.haim_4 = HAIM(512)
        self.haim_3 = HAIM(320)
        self.haim_2 = HAIM(128)
        self.haim_1 = HAIM(64)


        self.d_branch1 = BasicConv2d(1, 1, 3, padding=1, dilation=1)
        self.d_branch2 = BasicConv2d(1, 1, 3, padding=3, dilation=3)
        self.d_branch3 = BasicConv2d(1, 1, 3, padding=5, dilation=5)

    def forward_features(self, x, timesteps, cond_img, depth):
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0]))
        time_token = time_token.unsqueeze(dim=1)

        B = x.shape[0]
        outs = []


        depth1 = self.d_branch1(depth)
        depth2 = self.d_branch2(depth)
        depth3 = self.d_branch3(depth)
        # depth_1 = torch.cat([depth, depth], dim=1)
        x_depth = torch.cat([depth1, depth2, depth], dim=1)
        x_depth, H, W = self.patch_embed11(x_depth)
        # stage 1
        x, H, W = self.patch_embed1(cond_img, x)

        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)

        x = self.norm1(x)
        # time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_depth = torch.cat([time_token, x_depth], dim=1)
        for i, blk in enumerate(self.block1):
            x_depth = blk(x_depth, H, W)
        x_depth = self.norm1(x_depth)
        time_token = x[:, 0]
        x_depth = x_depth[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x1 = self.haim_1(x, x_depth)
        outs.append(x1)


        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 2
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        # time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_depth, H, W = self.patch_embed2(x_depth)
        x_depth = torch.cat([time_token, x_depth], dim=1)
        for i, blk in enumerate(self.block2):
            x_depth = blk(x_depth, H, W)
        x_depth = self.norm2(x_depth)
        time_token = x[:, 0]
        x_depth = x_depth[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x2 = self.haim_2(x, x_depth)

        outs.append(x2)

        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 3
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        # time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_depth, H, W = self.patch_embed3(x_depth)
        x_depth = torch.cat([time_token, x_depth], dim=1)
        for i, blk in enumerate(self.block3):
            x_depth = blk(x_depth, H, W)
        x_depth = self.norm3(x_depth)
        time_token = x[:, 0]
        x_depth = x_depth[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x3 = self.haim_3(x, x_depth)

        outs.append(x3)

        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        # time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_depth, H, W = self.patch_embed4(x_depth)
        x_depth = torch.cat([time_token, x_depth], dim=1)
        for i, blk in enumerate(self.block4):
            x_depth = blk(x_depth, H, W)
        x_depth = self.norm4(x_depth)
        time_token = x[:, 0]
        x_depth = x_depth[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x4 = self.haim_4(x, x_depth)

        outs.append(x4)

        return outs

    def forward(self, x, timesteps, cond_img, depth):
        x = self.forward_features(x, timesteps, cond_img, depth)

        #        x = self.head(x[3])

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here提取时间令牌：从输入张量 x 中选择第一个时间步的信息，将其形状变为 (B, 1, C)。
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


from timm.models.layers import DropPath
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class dedc(nn.Module):
    """
    FAF module
    """

    def __init__(self, dims=[64, 128, 320, 512], dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(dims[0], dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dims[1], dim, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(dims[2], dim, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(dims[3], dim, 3, stride=1, padding=1)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(dims[0], dim, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(dim), nn.PReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(dims[1], dim, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(dim), nn.PReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(dims[2], dim, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(dim), nn.PReLU()
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(dims[3], dim, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(dim), nn.PReLU()
        )
        self.conv2d = BasicConv2d(dim*2, dim, kernel_size=3, stride=1, padding=1)

        self.dct1 = DCTM(256, 11, 22)
        self.dct2 = DCTM(256, 22, 44)
        self.dct3 = DCTM(256, 44, 88)

        self.dc = ConvModule(in_channels=dim * 2, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))


    def forward(self, c1, c2, c3, c4):
        # x = self.proj(x)#(5,512,11,11) -> (5,256,11,11)(5,11*11,256) -> (5,256,11*11)
        n, _, h, w = c4.shape

        x1 = self.conv1(c1)
        x11 = self.Conv1(c1)
        x1 = torch.cat((x1, x11), 1)
        x1 = self.conv2d(x1)
        # x11 = self.conv11(x1)

        x2 = self.conv2(c2)
        x22 = self.Conv2(c2)
        x2 = torch.cat((x2, x22), 1)
        x2 = self.conv2d(x2)
        # x22 = self.conv22(x2)

        x3 = self.conv3(c3)
        x33 = self.Conv3(c3)
        x3 = torch.cat((x3, x33), 1)
        x3 = self.conv2d(x3)
        # x33 = self.conv33(x3)

        x4 = self.conv4(c4)
        x44 = self.Conv4(c4)
        x4 = torch.cat((x4, x44), 1)
        x4 = self.conv2d(x4)

        a1 = self.dct1(x4, x3)
        a2 = self.dct2(x3, x2)
        a3 = self.dct3(x2, x1)

        s1 = a3.flatten(2).transpose(1, 2)
        s1 = s1.permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        s2 = a2.flatten(2).transpose(1, 2)
        s2 = s2.permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        s2 = resize(s2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        s3 = a1.flatten(2).transpose(1, 2)
        s3 = s3.permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        s3 = resize(s3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        x = self.dc(torch.cat([s1, s2], dim=1))
        x = self.dc(torch.cat([s3, x], dim=1))

        return x

def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

class DCTM(nn.Module):
    """
    NMI module
    """
    def __init__(self, dim, in_w, in_w2, in_channel = 256):
        super(DCTM, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_w, np.int(in_w//4), bias=True)
        self.fc1 = nn.Linear(np.int(in_w//4), in_w, bias=True)

        self.fc2 = nn.Linear(in_w2, np.int(in_w2//4), bias=True)
        self.fc3 = nn.Linear(np.int(in_w2//4), in_w2, bias=True)

        self.relu = nn.ReLU()
        self.sam = SAM(dim)

        self.linear_fuse = ConvModule(in_channels=dim * 2, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

    def forward(self, x1, x2):

        n, _, h, w1 = x1.shape
        # n, _, h, w2 = x2.shape

        y1 = self.conv1(x1)
        y1 = self.sam(y1)

        z1 = dct.dct_2d(x1)
        z1 = self.fc(z1)
        z1 = self.relu(z1)
        z1 = self.fc1(z1)
        z1 =self.sigmoid(z1)

        c2 = resize(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)

        z1 = c2.mul(z1)

        y2 = self.conv1(x2)
        y2 = self.sam(y2)

        z2 = dct.dct_2d(x2)
        z2 = self.fc2(z2)
        z2 = self.relu(z2)
        z2 = self.fc3(z2)
        z2 =self.sigmoid(z2)


        c1 = resize(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)

        z2 = c1.mul(z2)

        a1 = z1 + y1
        a2 = z2 + y2

        s1 = a1.flatten(2).transpose(1, 2)
        s1 = s1.permute(0, 2, 1).reshape(n, -1, x1.shape[2], x1.shape[3])
        s1 = resize(s1, size=x2.size()[2:], mode='bilinear', align_corners=False)

        s2 = a2.flatten(2).transpose(1, 2)
        s2 = s2.permute(0, 2, 1).reshape(n, -1, x2.shape[2], x2.shape[3])


        x = self.linear_fuse(torch.cat([s1, s2], dim=1))


        return x

class SAM(nn.Module):
    def __init__(self, in_channel = 256):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)


        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)


        x_cat = torch.cat((MaxPool, AvgPool), dim=1)


        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x + x

        return x

class Decoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num
        embedding_dim = dim

        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(

            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

        self.dedc1 = dedc([64, 128, 320, 512], 256)


    def forward(self, inputs, timesteps, x):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        c1, c2, c3, c4 = inputs
        t1 = c1.size()

        n, _, h, w = c4.shape

        _x = [x]

        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)

        _c = self.dedc1(c1, c2, c3, c4)

        x = torch.cat([_c, x], dim=1)
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)

        x = self.pred(x) # 输出预测
        return x, c1, c2, c3, c4


class net(nn.Module):
    def __init__(self, class_num=2, mask_chans=0, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        self.backbone = pvt_v2_b4_m(in_chans=3, mask_chans=mask_chans)
        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=class_num, mask_chans=mask_chans)
        self._init_weights()  # load pretrain

    def forward(self, x, timesteps, cond_img, depth):
        features = self.backbone(x, timesteps, cond_img, depth)
        features, layer1, layer2, layer3, layer4 = self.decode_head(features, timesteps, x)
        return features

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0',
            'pvt_v2_b1',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b4_m',
            'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        pretrained_dict = torch.load(self._download_weights('pvt_v2_b4_m')) #for save mem
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img, depth):
        return self.forward(x, timesteps, cond_img, depth)

    def extract_features(self, cond_img):
        # do nothing
        return cond_img


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass