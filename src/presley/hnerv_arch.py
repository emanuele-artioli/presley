"""HNeRV (Hybrid Neural Representation for Videos, arXiv 2304.02633) architecture.

Learned-codec comparison baseline for `reports/9_codec_baselines_report.md`
(reviewer-critical gap R2/R5, `reports/6_action_matrix.md`). HNeRV encodes a
video as the *weights* of a small implicit network overfit to that one clip:

- "Encoding" = a per-video training run that jointly learns a lightweight
  content-adaptive CNN encoder (the "H" in HNeRV — hybrid between NeRV's pure
  index/positional embedding and a full autoencoder) and a decoder built from
  PixelShuffle upsampling blocks ("NeRV blocks").
- "Decoding" = a single forward pass of the *decoder only* through the
  per-frame embeddings produced by the encoder during training. The encoder
  itself is discarded after training (never transmitted) — only the decoder
  weights and the quantized per-frame embeddings need to reach the client,
  matching how HNeRV's compression numbers are reported in the paper.

This module intentionally simplifies the paper's ConvNeXt-based encoder to a
plain strided-conv "patchify" stack (kernel_size == stride, no overlap) — a
lightweight content-adaptive embedding in the same spirit, not a literal
reimplementation of the paper's exact encoder blocks. The decoder's NeRV
blocks (conv + PixelShuffle + activation) follow the paper's core mechanism
directly.

No pretrained HNeRV weights or reference implementation were found under
`assets/weights/` (searched `/home/itec/emanuele/Models` first, per
CLAUDE.md's weights convention) — this is a from-scratch implementation,
trained per-video via `scripts/hnerv_baseline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HNeRVConfig:
    """Static shape contract for one HNeRV encoder/decoder pair.

    `strides` are the decoder's per-stage PixelShuffle upscale factors,
    applied in order starting from the embedding grid; their product must
    exactly divide (height, width) down to (embed_height, embed_width). The
    encoder mirrors the same strides in reverse as plain non-overlapping
    strided convolutions.
    """

    height: int
    width: int
    embed_height: int
    embed_width: int
    embed_channels: int
    strides: tuple[int, ...]
    channels: tuple[int, ...]  # decoder output channels per stage, len(channels) == len(strides)

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.strides):
            raise ValueError(
                f"channels ({len(self.channels)}) and strides ({len(self.strides)}) must have equal length"
            )
        product = math.prod(self.strides) if self.strides else 1
        if self.embed_height * product != self.height:
            raise ValueError(
                f"embed_height ({self.embed_height}) * stride product ({product}) "
                f"!= height ({self.height})"
            )
        if self.embed_width * product != self.width:
            raise ValueError(
                f"embed_width ({self.embed_width}) * stride product ({product}) "
                f"!= width ({self.width})"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "embed_height": self.embed_height,
            "embed_width": self.embed_width,
            "embed_channels": self.embed_channels,
            "strides": list(self.strides),
            "channels": list(self.channels),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "HNeRVConfig":
        return HNeRVConfig(
            height=int(payload["height"]),
            width=int(payload["width"]),
            embed_height=int(payload["embed_height"]),
            embed_width=int(payload["embed_width"]),
            embed_channels=int(payload["embed_channels"]),
            strides=tuple(int(s) for s in payload["strides"]),
            channels=tuple(int(c) for c in payload["channels"]),
        )


class NeRVBlock(nn.Module):
    """Conv -> PixelShuffle -> GELU upsampling block (the core HNeRV/NeRV unit)."""

    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor**2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [Batch, InChannels, H, W]
        x = self.conv(x)  # Shape: [Batch, OutChannels * r^2, H, W]
        x = self.pixel_shuffle(x)  # Shape: [Batch, OutChannels, H*r, W*r]
        return self.act(x)


class HNeRVDecoder(nn.Module):
    """Stack of NeRVBlocks mapping a per-frame embedding grid up to an RGB frame."""

    def __init__(self, config: HNeRVConfig) -> None:
        super().__init__()
        self.config = config
        blocks = []
        in_channels = config.embed_channels
        for stride, out_channels in zip(config.strides, config.channels):
            blocks.append(NeRVBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        # embedding Shape: [Batch, EmbedChannels, EmbedHeight, EmbedWidth]
        x = embedding
        for block in self.blocks:
            x = block(x)
        x = self.head(x)  # Shape: [Batch, 3, Height, Width]
        return torch.sigmoid(x)


class HNeRVEncoder(nn.Module):
    """Lightweight content-adaptive embedding: mirrors the decoder's strides in reverse.

    Each stage is a non-overlapping strided conv (kernel_size == stride) — a
    simplified stand-in for the paper's ConvNeXt encoder, not a literal port.
    """

    def __init__(self, config: HNeRVConfig) -> None:
        super().__init__()
        strides = list(reversed(config.strides))
        # Decoder stage outputs, reversed, shifted so the encoder's final stage
        # projects down to embed_channels.
        decoder_channels_reversed = list(reversed(config.channels))
        out_channels_seq = decoder_channels_reversed[1:] + [config.embed_channels]

        layers: list[nn.Module] = []
        in_channels = 3
        for stride, out_channels in zip(strides, out_channels_seq):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, padding=0))
            layers.append(nn.GELU())
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames Shape: [Batch, 3, Height, Width], values in [0, 1]
        return self.net(frames)  # Shape: [Batch, EmbedChannels, EmbedHeight, EmbedWidth]


class HNeRVModel(nn.Module):
    """Joint encoder+decoder used only during training; decode-time uses HNeRVDecoder alone."""

    def __init__(self, config: HNeRVConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = HNeRVEncoder(config)
        self.decoder = HNeRVDecoder(config)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # frames Shape: [Batch, 3, Height, Width]
        embedding = self.encoder(frames)  # Shape: [Batch, EmbedChannels, EmbedHeight, EmbedWidth]
        reconstruction = self.decoder(embedding)  # Shape: [Batch, 3, Height, Width]
        return reconstruction, embedding


def count_decoder_parameters(decoder: HNeRVDecoder) -> int:
    return int(sum(p.numel() for p in decoder.parameters()))


def quantize_tensor_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Per-tensor affine int8 quantization. Returns (quantized_uint8, scale, zero_point)."""
    t_min = float(tensor.min().item())
    t_max = float(tensor.max().item())
    scale = (t_max - t_min) / 255.0 if t_max > t_min else 1.0
    zero_point = t_min
    quantized = torch.clamp(torch.round((tensor - zero_point) / scale), 0, 255).to(torch.uint8)
    return quantized, scale, zero_point


def dequantize_tensor_int8(quantized: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    return quantized.to(torch.float32) * scale + zero_point


def save_hnerv_checkpoint(path: str | Path, decoder: HNeRVDecoder, embeddings: torch.Tensor) -> int:
    """Serialize decoder (fp16) + quantized (int8) embeddings, gzip-compressed.

    Only the decoder and per-frame embeddings are saved — the encoder is
    encode-time-only and is never part of the transmitted "bytes" for this
    baseline, matching how HNeRV's own compression accounting works.

    Returns the size in bytes of the file written to `path`.
    """
    quantized, scale, zero_point = quantize_tensor_int8(embeddings)
    decoder_state_fp16 = {key: value.half().cpu() for key, value in decoder.state_dict().items()}
    payload = {
        "config": decoder.config.as_dict(),
        "decoder_state_dict": decoder_state_fp16,
        "embeddings_int8": quantized.cpu(),
        "embed_scale": scale,
        "embed_zero_point": zero_point,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    compressed = gzip.compress(buffer.getvalue(), compresslevel=9)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(compressed)
    return len(compressed)


def load_hnerv_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[HNeRVDecoder, torch.Tensor]:
    """Inverse of `save_hnerv_checkpoint`: rebuilds the decoder and dequantizes embeddings.

    This is the "decode" side of the Residual-Guarantee-style symmetry check:
    a real client would load exactly this file and run exactly this forward
    pass — no information beyond what was written to disk is used.
    """
    compressed = Path(path).read_bytes()
    raw = gzip.decompress(compressed)
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    config = HNeRVConfig.from_dict(payload["config"])
    decoder = HNeRVDecoder(config)
    decoder.load_state_dict({key: value.float() for key, value in payload["decoder_state_dict"].items()})
    decoder = decoder.to(device)
    embeddings = dequantize_tensor_int8(
        payload["embeddings_int8"].to(device), payload["embed_scale"], payload["embed_zero_point"]
    )
    return decoder, embeddings
