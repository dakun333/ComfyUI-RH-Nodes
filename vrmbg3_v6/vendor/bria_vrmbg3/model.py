"""VRMBG-3.0 inference model — self-contained, no internal dependencies.

Architecture: backbone_6ch variant (6-channel Swin-L BiRefNet).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.ops import deform_conv2d
from transformers import PreTrainedModel

from .vrmbg3_config import VRMBG3Config

# ── fixed architecture constants ─────────────────────────────────────────────
_IN_CH      = 6
_LAT_CH     = [3072, 1536, 768, 384]   # swin_v1_l channels after cat-pyramid doubling
_CXT        = [384, 768, 1536]          # multi-scale skip channels fed into deepest feature


# ── Swin-V1-L backbone ────────────────────────────────────────────────────────

class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    C = int(windows.shape[-1])
    x = windows.view(-1, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)


class _WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size   = window_size
        self.num_heads     = num_heads
        self.scale         = qk_scale or (dim // num_heads) ** -0.5
        self.attn_drop_prob = attn_drop

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        ch = torch.arange(window_size[0])
        cw = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([ch, cw], indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).unsqueeze(0).to(q)
        attn_mask = rpb
        if mask is not None:
            attn_mask = attn_mask + mask.to(dtype=q.dtype).unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop_prob if self.training else 0.0)
        x = out.transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class _SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size
        assert 0 <= shift_size < window_size
        self.norm1    = norm_layer(dim)
        self.attn     = _WindowAttention(dim, to_2tuple(window_size), num_heads,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2    = norm_layer(dim)
        self.mlp      = _Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.H = self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W    = self.H, self.W
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            nW, N  = mask_matrix.size(0), mask_matrix.size(1)
            am     = mask_matrix.unsqueeze(0).expand(B, nW, N, N).reshape(B * nW, N, N)
        else:
            shifted, am = x, None
        xw  = _window_partition(shifted, self.window_size).view(-1, self.window_size ** 2, C)
        aw  = self.attn(xw, mask=am).view(-1, self.window_size, self.window_size, C)
        shifted = _window_reverse(aw, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim       = dim
        self.norm      = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class _BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size   = window_size
        self.shift_size    = window_size // 2
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            _SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                       shift_size=0 if i % 2 == 0 else window_size // 2,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, H, W):
        ws = self.window_size
        ss = self.shift_size
        Hp = torch.ceil(torch.tensor(H) / ws).to(torch.int64) * ws
        Wp = torch.ceil(torch.tensor(W) / ws).to(torch.int64) * ws
        img_mask = x.new_zeros((1, Hp, Wp, 1))
        for hi, h in enumerate((slice(0, -ws), slice(-ws, -ss), slice(-ss, None))):
            for wi, w in enumerate((slice(0, -ws), slice(-ws, -ss), slice(-ss, None))):
                img_mask[:, h, w, :] = hi * 3 + wi
        mw  = _window_partition(img_mask, ws).view(-1, ws * ws)
        am  = (mw.unsqueeze(1) - mw.unsqueeze(2)).masked_fill(
            (mw.unsqueeze(1) - mw.unsqueeze(2)) != 0, float('-inf')
        ).masked_fill((mw.unsqueeze(1) - mw.unsqueeze(2)) == 0, 0.0).to(x.dtype)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = checkpoint.checkpoint(blk, x, am) if self.use_checkpoint else blk(x, am)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            return x, H, W, x_down, (H + 1) // 2, (W + 1) // 2
        return x, H, W, x, H, W


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(
                -1, self.proj.out_channels, Wh, Ww)
        return x


class _SwinTransformer(nn.Module):
    def __init__(self, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, in_channels=3, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_layers  = len(depths)
        self.out_indices = out_indices
        self.patch_embed = _PatchEmbed(4, in_channels, embed_dim, norm_layer if patch_norm else None)
        self.pos_drop    = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            _BasicLayer(
                dim=int(embed_dim * 2 ** i), depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=_PatchMerging if i < self.num_layers - 1 else None)
            for i in range(self.num_layers)])
        for i in out_indices:
            self.add_module(f'norm{i}', norm_layer(int(embed_dim * 2 ** i)))

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        outs = []
        for i, layer in enumerate(self.layers):
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                x_out = getattr(self, f'norm{i}')(x_out)
                outs.append(
                    x_out.view(-1, H, W, int(self.embed_dim * 2 ** i)).permute(0, 3, 1, 2).contiguous())
        return tuple(outs)


def _swin_v1_l(in_channels: int = 3) -> _SwinTransformer:
    return _SwinTransformer(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                            window_size=12, in_channels=in_channels)


# ── Deformable convolution ────────────────────────────────────────────────────

class _DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride   = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding  = padding
        self.offset_conv    = nn.Conv2d(in_channels, 2 * ks[0] * ks[1], ks, stride, padding, bias=True)
        self.modulator_conv = nn.Conv2d(in_channels,     ks[0] * ks[1], ks, stride, padding, bias=True)
        self.regular_conv   = nn.Conv2d(in_channels, out_channels,       ks, stride, padding, bias=bias)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

    def forward(self, x):
        offset    = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        return deform_conv2d(x, offset, self.regular_conv.weight, self.regular_conv.bias,
                             padding=self.padding, mask=modulator, stride=self.stride)


# ── ASPPDeformable ────────────────────────────────────────────────────────────

class _ASPPModDeform(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = _DeformableConv2d(in_channels, planes, kernel_size, 1, padding, bias=False)
        self.bn   = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class _ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        inter = 256
        self.aspp1       = _ASPPModDeform(in_channels, inter, 1, 0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModDeform(in_channels, inter, s, s // 2) for s in parallel_block_sizes])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True))
        self.conv1    = nn.Conv2d(inter * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.dropout  = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        xs = [d(x) for d in self.aspp_deforms]
        x5 = F.interpolate(self.global_avg_pool(x), size=x1.size()[2:], mode='bilinear', align_corners=True)
        x  = torch.cat([x1, *xs, x5], 1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


# ── Decoder building blocks ───────────────────────────────────────────────────

class _BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        inter        = 64
        self.conv_in = nn.Conv2d(in_channels, inter, 3, 1, 1)
        self.bn_in   = nn.BatchNorm2d(inter)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = _ASPPDeformable(in_channels=inter)
        self.conv_out = nn.Conv2d(inter, out_channels, 3, 1, 1)
        self.bn_out   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        x = self.dec_att(x)
        return self.bn_out(self.conv_out(x))


class _BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)


class _SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


# ── Decoder ───────────────────────────────────────────────────────────────────

def _img2patches(image: torch.Tensor, patch_ref: torch.Tensor) -> torch.Tensor:
    gh = image.shape[-2] // patch_ref.shape[-2]
    gw = image.shape[-1] // patch_ref.shape[-1]
    return rearrange(image, 'b c (hg h) (wg w) -> b (c hg wg) h w', hg=gh, wg=gw)


class _Decoder(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        ipt_out = [channels[i] // 8 for i in range(4)]   # [384, 192, 96, 48]
        ic = 64
        # input-pyramid blocks (split=True: patches carry c*grid² channels)
        self.ipt_blk5 = _SimpleConvs(2**10 * 3, ipt_out[0], ic)  # 3072 → 384
        self.ipt_blk4 = _SimpleConvs(2**8  * 3, ipt_out[0], ic)  # 768  → 384
        self.ipt_blk3 = _SimpleConvs(2**6  * 3, ipt_out[1], ic)  # 192  → 192
        self.ipt_blk2 = _SimpleConvs(2**4  * 3, ipt_out[2], ic)  # 48   → 96
        self.ipt_blk1 = _SimpleConvs(3,          ipt_out[3], ic)  # 3    → 48

        dec_out = [channels[1], channels[2], channels[3], channels[3] // 2]  # [1536,768,384,192]
        dec_in  = [
            channels[0] + ipt_out[0],  # 3072+384=3456
            channels[1] + ipt_out[0],  # 1536+384=1920
            channels[2] + ipt_out[1],  #  768+192=960
            channels[3] + ipt_out[2],  #  384+96 =480
        ]
        self.decoder_block4 = _BasicDecBlk(dec_in[0], dec_out[0])
        self.decoder_block3 = _BasicDecBlk(dec_in[1], dec_out[1])
        self.decoder_block2 = _BasicDecBlk(dec_in[2], dec_out[2])
        self.decoder_block1 = _BasicDecBlk(dec_in[3], dec_out[3])
        self.conv_out1      = nn.Sequential(nn.Conv2d(dec_out[3] + ipt_out[3], 1, 1))  # 192+48=240

        self.lateral_block4 = _BasicLatBlk(channels[1], dec_out[0])
        self.lateral_block3 = _BasicLatBlk(channels[2], dec_out[1])
        self.lateral_block2 = _BasicLatBlk(channels[3], dec_out[2])

        # multi-scale supervision heads (weights in checkpoint; attn gates used at inference)
        self.conv_ms_spvn_4 = nn.Conv2d(dec_out[0], 1, 1)
        self.conv_ms_spvn_3 = nn.Conv2d(dec_out[1], 1, 1)
        self.conv_ms_spvn_2 = nn.Conv2d(dec_out[2], 1, 1)
        _N = 16
        self.gdt_convs_4      = nn.Sequential(nn.Conv2d(dec_out[0], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_3      = nn.Sequential(nn.Conv2d(dec_out[1], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_2      = nn.Sequential(nn.Conv2d(dec_out[2], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1))
        self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1))
        self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1))
        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1))

    def _ipt(self, x, ref, blk):
        patches = _img2patches(x, ref)
        return blk(F.interpolate(patches, size=ref.shape[2:], mode='bilinear', align_corners=True))

    def forward(self, features: list) -> list:
        x, x1, x2, x3, x4 = features

        x4 = torch.cat((x4, self._ipt(x, x4, self.ipt_blk5)), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p3 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True) \
              + self.lateral_block4(x3)

        _p3 = torch.cat((_p3, self._ipt(x, _p3, self.ipt_blk4)), 1)
        p3  = self.decoder_block3(_p3)
        p3  = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        _p2 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True) \
              + self.lateral_block3(x2)

        _p2 = torch.cat((_p2, self._ipt(x, _p2, self.ipt_blk3)), 1)
        p2  = self.decoder_block2(_p2)
        p2  = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        _p1 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True) \
              + self.lateral_block2(x1)

        _p1 = torch.cat((_p1, self._ipt(x, _p1, self.ipt_blk2)), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)

        _p1 = torch.cat((_p1, self._ipt(x, _p1, self.ipt_blk1)), 1)
        return [self.conv_out1(_p1)]


# ── BiRefNet ──────────────────────────────────────────────────────────────────

class BiRefNet(PreTrainedModel):
    config_class = VRMBG3Config

    def __init__(self, config=None):
        if config is None:
            config = VRMBG3Config()
        super().__init__(config)
        self.bb             = _swin_v1_l(in_channels=_IN_CH)
        self.squeeze_module = nn.Sequential(_BasicDecBlk(_LAT_CH[0] + sum(_CXT), _LAT_CH[0]))
        self.decoder        = _Decoder(_LAT_CH)

    def _enc(self, x: torch.Tensor):
        x1, x2, x3, x4 = self.bb(x)
        # cat-pyramid: run backbone again at half resolution and concatenate
        B, C, H, W = x.shape
        xp = F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
        x1_, x2_, x3_, x4_ = self.bb(xp)
        x1 = torch.cat([x1, F.interpolate(x1_, x1.shape[2:], mode='bilinear', align_corners=True)], 1)
        x2 = torch.cat([x2, F.interpolate(x2_, x2.shape[2:], mode='bilinear', align_corners=True)], 1)
        x3 = torch.cat([x3, F.interpolate(x3_, x3.shape[2:], mode='bilinear', align_corners=True)], 1)
        x4 = torch.cat([x4, F.interpolate(x4_, x4.shape[2:], mode='bilinear', align_corners=True)], 1)
        # multi-scale context: concat x1,x2,x3 (all scales) into x4
        x4 = torch.cat([
            F.interpolate(x1, x4.shape[2:], mode='bilinear', align_corners=True),
            F.interpolate(x2, x4.shape[2:], mode='bilinear', align_corners=True),
            F.interpolate(x3, x4.shape[2:], mode='bilinear', align_corners=True),
            x4,
        ], 1)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x1, x2, x3, x4 = self._enc(x)
        x4 = self.squeeze_module(x4)
        return self.decoder([x[:, :3], x1, x2, x3, x4])


def build_vrmbg3() -> BiRefNet:
    """Instantiate VRMBG-3.0. Load weights separately with model.load_state_dict(...)."""
    return BiRefNet()
