## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

import math
import copy
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import sys
sys.path.append('/zs/Restormer/basicsr/')
from net.DegEncoder import DeepDegradationEncoder

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SS2D_1(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        require_prompt=False,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        softmax_version=False,
        forward_type="v2",
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        #d_model model dim
        d_expand = int(ssm_ratio * d_model)
        #d_inner  dim in model, for channel it should be 2
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.softmax_version = softmax_version
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv

        #cccc
        dc_inner = 4 
        self.dtc_rank = 6 #6
        self.dc_state = 16 #16
        self.conv_cin = nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = nn.Conv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.forward_core=self.forward_corev1
        # self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K = 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1        
        self.KC = 2
        self.K2C = self.KC if forward_type not in ["share_a"] else 1
        self.cforward_core = self.cforward_corev1
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')
        # in proj =======================================
        self.in_conv = nn.Conv2d(in_channels=d_model, out_channels=d_expand * 2, kernel_size=1, stride=1, padding=0)
        self.act: nn.Module = act_layer()
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        self.out_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        # xc proj ============================
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0)) # (K, N, inner)
        del self.xc_proj


        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_conv = nn.Conv2d(in_channels=d_expand, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Dsc = nn.Parameter(torch.ones((self.K2C * dc_inner)))
        self.Ac_logs = nn.Parameter(torch.randn((self.K2C * dc_inner, self.dc_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner))) 
        self.require_prompt = require_prompt
        if self.require_prompt:
            self.prompt_delta = nn.Sequential(
                nn.Linear(in_features=512, out_features=d_inner//16),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=d_inner//16, out_features=d_inner)
                )

            self.prompt_B = nn.Linear(in_features=512, out_features=self.d_state)
            self.prompt_C = nn.Linear(in_features=512, out_features=self.d_state)

        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj


    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log


    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    

    
    def forward_corev1(self, x: torch.Tensor, emb):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        # x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1) #一个h,w展开，一个w,h展开，然后堆在一起
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l) #把上面那俩再翻译下，然后堆在一起
        xs = x.view(B, 1, -1, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        if self.require_prompt:
            prompt_B = torch.sigmoid(self.prompt_B(emb.view(B, 512))).view(B, 1, -1, 1)
            prompt_delta = torch.sigmoid(self.prompt_delta(emb.view(B, 512))).view(B, -1, 1)
            prompt_C = torch.sigmoid(self.prompt_C(emb.view(B, 512))).view(B, 1, -1, 1)

            # prompt_B = 1
            # prompt_delta = 1
            dts = (dts*prompt_delta).contiguous()
            Bs = (Bs * prompt_B).contiguous()
            Cs = (Cs * prompt_C).contiguous()
        # linear
        # prompt有一定的输入依赖，和xs关联
        # 压缩、重复、尽可能简单
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, self.K, -1, L)
        # assert out_y.dtype == torch.float16


        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        y = out_y
        y = y.view(B, C, H, W)
        y = self.out_norm(y).to(x.dtype)
        
        return y

    def cforward_corev1(self, xc: torch.Tensor):
        self.selective_scanC = selective_scan_fn

        b,d,h,w = xc.shape
        
        #xc = self.pooling(xc).squeeze(-1).permute(0,2,1).contiguous() #b,1,d, >1!
        #print("xc shape", xc.shape) # 8,1,96
        xc = self.pooling(xc) #b,d,1,1
        low_freq = xc
        xc = xc.permute(0,2,1,3).contiguous() #b,1,d,1
        xc = self.conv_cin(xc) #b,4,d,1
        xc = xc.squeeze(-1) #b,4,d

        #xc = xc.permute(0,2,1).contiguous()
        #xc = self.conv_innerc(xc)
        #xc = xc.permute(0,2,1).contiguous()

        B, D, L = xc.shape #b,1,c
        D, N = self.Ac_logs.shape #2,16
        K, D, R = self.dtc_projs_weight.shape #2,1,6

        #print("Ac_logs shape", self.Ac_logs.shape) #2,16
        #print("dtc_projs_weight shape", self.dtc_projs_weight.shape) #2,1,6
        #print("xc_proj_weight shape", self.xc_proj_weight.shape) #2,38,1
        #print("Dsc shape", self.Dsc.shape) # 2
        #print("dtc_projs_bias shape", self.dtc_projs_bias.shape) #2,1

        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1) #input:b,d,l output:b,2,d,l
        #print("xsc shape", xsc.shape) # 8,2,1,96

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight) #8,2,1,96; 2,38,1 ->8,2,38,96
        
        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2) # 8,2,38,96-> 6,16,16
        #dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()

        xsc = xsc.view(B, -1, L) # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous() # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc # (k * d) 2 
        dt_projs_bias = self.dtc_projs_bias.view(-1) # (k * d)2

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16
        

        #print("channel xs shape", xsc.shape) #8, 2, 192
        #print("channel dts shape", dts.shape)#8, 2, 192
        #print("channel As shape", As.shape)#2, 16
        #print("channel Bs shape", Bs.shape)#8, 2, 16, 192
        #print("channel Cs shape", Cs.shape)#8, 2, 16, 192
        #print("channel Ds shape", Ds.shape)#2
        #print("channel dt_projs_bias shape", dt_projs_bias.shape) #2
        out_y = self.selective_scanC(
            xsc, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()
        #y = xsc[:, 0].float() + torch.flip(xsc[:, 1], dims=[-1]).float()


        #y: b,4,d
        y = y.unsqueeze(-1) # b,4,d,1
        y = self.conv_cout(y) # b,1,d,1
        y = y.transpose(dim0=1, dim1=2).contiguous() # b,d,1,1
        y = self.channel_norm(y)
        y = y.to(xc.dtype)

        #y = y.transpose(dim0=1, dim1=2).contiguous().unsqueeze(-1).contiguous()
        #y = self.channel_norm(y)
        #y = y.to(xc.dtype)

        #y = y.permute(0,2,1,3).contiguous()
        #y = self.conv_outc(y)
        #y = y.permute(0,2,1,3).contiguous()
        
        return y, low_freq


    def forward(self, x: torch.Tensor, emb, **kwargs):
        #input: b,d,h,w
        #output: b,d,h,w
        xz = self.in_conv(x)
        x, z = xz.chunk(2, dim=1) # (b, d, h, w)
        if not self.softmax_version:
            z = self.act(z)
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        y1 = self.forward_core(x, emb)
        y2 = y1 * z
        c, low_freq = self.cforward_core(y2)#x:b,d,h,w; output:b,d,1,1
        # c = 1
        y2 = y2 + c + self.alpha * (y2 - low_freq)
        out = self.out_conv(y2)
        return out


##########################################################################
class MamberBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, require_prompt=False):
        super(MamberBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SS2D_1(d_model=dim, ssm_ratio=1, require_prompt=require_prompt)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.se = SELayer(channel=dim)

    def forward(self, x, emb=None):
        x = x + self.attn(self.norm1(x), emb)
        # x = self.se(x)
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Mamber -----------------------
class MultiInputSequential(nn.Module):
    def __init__(self, *layers):
        super(MultiInputSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, emb, *args):
        # 依次将 (x, *args) 传递给每个子模块
        for layer in self.layers:
            x = layer(x, emb, *args)
        return x

class DPMambaIR(nn.Module):
    def __init__(self, 
                 inp_channels=3, 
                 out_channels=3, 
                 dim=48,
                 num_blocks=[4, 6, 6, 8], 
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  # 另一种选择 'BiasFree'
                 dual_pixel_task=False      # True 用于双像素去模糊任务，同时设置 inp_channels=6
                ):
        super(DPMambaIR, self).__init__()

        # 输入特征提取
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.mlp_deg = nn.Conv2d(512, 256, 1, 1, 0)

        self.proj_to256 = nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_todim = nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, stride=1, padding=1)

        
        # 编码器部分
        self.encoder_level1 = MultiInputSequential(
            *[MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=True) 
              for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim)  # 从 Level 1 到 Level 2
        self.encoder_level2 = MultiInputSequential(
            *[MamberBlock(dim=dim * 2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=True) 
              for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2)  # 从 Level 2 到 Level 3
        self.encoder_level3 = MultiInputSequential(
            *[MamberBlock(dim=dim * 4, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=True) 
              for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4)  # 从 Level 3 到 Level 4

        # 中间特征提取（latent）
        self.latent = MultiInputSequential(
            *[MamberBlock(dim=dim * 8, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=True) 
              for _ in range(num_blocks[3])]
        )

        # 解码器部分
        self.up4_3 = Upsample(dim * 8)  # 从 Level 4 到 Level 3
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = MultiInputSequential(
            *[MamberBlock(dim=dim * 4, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=True) 
              for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)  # 从 Level 3 到 Level 2
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = MultiInputSequential(
            *[MamberBlock(dim=dim * 2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=True) 
              for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)  # 从 Level 2 到 Level 1
        self.decoder_level1 = MultiInputSequential(
            *[MamberBlock(dim=dim * 2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=True) 
              for _ in range(num_blocks[0])]
        )

        # 精细化部分
        self.refinement = MultiInputSequential(
            *[MamberBlock(dim=dim * 2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                          bias=bias, LayerNorm_type=LayerNorm_type, require_prompt=False) 
              for _ in range(num_refinement_blocks)]
        )

        # 输出部分
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # 双像素任务（可选）
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        # 加载退化编码器
        self.deg_encoder = DeepDegradationEncoder()
        self._load_pretrained_weights('/zs/DegAE_DegradationAutoencoder/experiments/Train_LQ_Reconstruct_All_addCls/models/50000_DDG_Encoder.pth')

    def _load_pretrained_weights(self, path):
        """加载预训练权重"""
        try:
            self.deg_encoder.load_state_dict(torch.load(path), strict=True)
            for param in self.deg_encoder.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Failed to load pretrained weights from {path}: {e}")
        

    def forward(self, inp_img):
        # 编码退化信息
        emb = self.deg_encoder(inp_img) # b 512 1 1
        z = self.mlp_deg(emb)
        # B, C, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)

        inp_enc_level1 = self.proj_to256(inp_enc_level1) * torch.sigmoid(z)
        inp_enc_level1 = self.proj_todim(inp_enc_level1)

        # 编码器
        out_enc_level1 = self.encoder_level1(inp_enc_level1, emb)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, emb)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, emb)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4, emb)

        # 解码器
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, emb)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, emb)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, emb)

        # 精细化
        out_dec_level1 = self.refinement(out_dec_level1, emb)

        # 双像素任务的特殊处理
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)

        # 输出生成
        out_dec_level1 = self.output(out_dec_level1) + inp_img if not self.dual_pixel_task else out_dec_level1

        return out_dec_level1


    def flops(self, shape=(3, 256, 256)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        #return sum(Gflops.values()) * 1e9
        return f"params(M) {params/1e6} GFLOPs {sum(Gflops.values())}"
    
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs):
    # print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


if __name__ == "__main__":
    net = DPMambaIR().to('cuda')
    print(net.flops())
    # import time
    # from tqdm import tqdm
    # def test_inference_speed(net, input_size=(1, 3, 224, 224), num_iterations=100):
    #     device = next(net.parameters()).device
    #     net.eval()
    #     input_tensor = torch.randn(*input_size).to(device)
    #     start_time = time.time()
    #     with torch.no_grad():
    #         for _ in tqdm(range(num_iterations)):
    #             _ = net(input_tensor)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     avg_time_per_iteration = elapsed_time / num_iterations
    #     print(f"平均每次推理时间: {avg_time_per_iteration * 1000:.3f} 毫秒")
    # test_inference_speed(net)