"""Self-contained NAFNet architecture (Chen et al., arXiv:2204.04676).

Vendored from megvii-research/NAFNet so we do not install the megvii
basicsr fork or any OpenCV-ONNX path. Inference only — no Local_Base /
training utilities. Copyright (c) 2022 megvii-model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel: int = 3,
        width: int = 16,
        middle_blk_num: int = 1,
        enc_blk_nums=None,
        dec_blk_nums=None,
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = []
        if dec_blk_nums is None:
            dec_blk_nums = []

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


# ---------------------------------------------------------------------------
# Local AvgPool replacement (TLSC / NAFNetLocal). Official GoPro *test* yml
# uses NAFNetLocal so AdaptiveAvgPool2d is swapped for a train-size-scaled
# window — global pool at 640×360 ≠ the 256 train crop and can artefact.
# Vendored from megvii-research/NAFNet basicsr/models/archs/local_arch.py.
# ---------------------------------------------------------------------------
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1 = min(h - 1, self.kernel_size[0] // r1)
                k2 = min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (
                    k1 * k2
                )
                out = F.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = F.pad(s, (1, 0, 1, 0))
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = (s4 + s1 - s2 - s3) / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = F.pad(out, pad2d, mode="replicate")
        return out


def replace_adaptive_pools(model: nn.Module, base_size, train_size, fast_imp: bool = False) -> None:
    for name, child in model.named_children():
        if len(list(child.children())) > 0:
            replace_adaptive_pools(child, base_size, train_size, fast_imp)
        if isinstance(child, nn.AdaptiveAvgPool2d):
            if child.output_size != 1:
                raise ValueError("NAFNet Local convert expects AdaptiveAvgPool2d(output_size=1)")
            setattr(
                model,
                name,
                AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size),
            )


def convert_to_local(
    model: NAFNet,
    *,
    train_size=(1, 3, 256, 256),
    base_size=None,
    fast_imp: bool = False,
) -> NAFNet:
    """In-place TLSC convert matching megvii NAFNetLocal.__init__."""
    N, C, H, W = train_size
    if base_size is None:
        base_size = (int(H * 1.5), int(W * 1.5))
    replace_adaptive_pools(model, base_size=base_size, train_size=train_size, fast_imp=fast_imp)
    model.eval()
    with torch.no_grad():
        model(torch.rand(train_size))
    return model


# GoPro deblur presets matching megvii options/test/GoPro/NAFNet-width{32,64}.yml
GOPRO_WIDTH64 = dict(
    width=64,
    enc_blk_nums=[1, 1, 1, 28],
    middle_blk_num=1,
    dec_blk_nums=[1, 1, 1, 1],
)
GOPRO_WIDTH32 = dict(
    width=32,
    enc_blk_nums=[1, 1, 1, 28],
    middle_blk_num=1,
    dec_blk_nums=[1, 1, 1, 1],
)


def build_gopro_nafnet(width: int = 64, *, local: bool = True) -> NAFNet:
    """Instantiate the GoPro deblur NAFNet matching the official checkpoint.

    ``local=True`` (default) applies the NAFNetLocal TLSC convert used by the
    official GoPro *test* yml — required for full-frame inference at sizes
    other than the 256 train crop.
    """
    if width == 64:
        cfg = GOPRO_WIDTH64
    elif width == 32:
        cfg = GOPRO_WIDTH32
    else:
        raise ValueError(f"Unsupported NAFNet GoPro width {width}; expected 32 or 64.")
    net = NAFNet(img_channel=3, **cfg)
    if local:
        convert_to_local(net)
    return net
