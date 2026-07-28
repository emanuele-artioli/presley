"""Fast-tier tests for the vendored NAFNet architecture (no weights, no GPU).

CI installs CPU torch; these only need torch + the self-contained
`presley.nafnet_arch` module — they must not import `presley.restoration`
(which pulls in InstantIR at module level).
"""
import torch
import torch.nn as nn

from presley.nafnet_arch import (
    AvgPool2d,
    NAFNet,
    build_gopro_nafnet,
    GOPRO_WIDTH64,
    GOPRO_WIDTH32,
)


def test_build_gopro_width64_matches_official_preset():
    net = build_gopro_nafnet(64, local=False)
    assert isinstance(net, NAFNet)
    assert net.intro.out_channels == GOPRO_WIDTH64["width"]
    assert len(net.encoders) == 4
    assert len(net.decoders) == 4


def test_build_gopro_width32_matches_official_preset():
    net = build_gopro_nafnet(32, local=False)
    assert net.intro.out_channels == GOPRO_WIDTH32["width"]


def test_build_gopro_rejects_unknown_width():
    try:
        build_gopro_nafnet(48)
    except ValueError as exc:
        assert "48" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported width")


def test_nafnet_forward_preserves_spatial_size():
    """check_image_size pads internally; the returned crop must match input HxW."""
    net = build_gopro_nafnet(32, local=False)
    net.eval()
    x = torch.rand(1, 3, 40, 56)  # not divisible by padder_size=16
    with torch.no_grad():
        y = net(x)
    assert y.shape == x.shape


def test_convert_to_local_replaces_adaptive_avg_pool():
    """Official GoPro test yml uses NAFNetLocal (TLSC); default build must
    swap AdaptiveAvgPool2d → AvgPool2d so full-frame SCA matches train crops."""
    net = build_gopro_nafnet(32, local=True)
    pools = [m for m in net.modules() if isinstance(m, AvgPool2d)]
    adaptive = [m for m in net.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    assert pools, "expected Local AvgPool2d replacements"
    assert not adaptive, "AdaptiveAvgPool2d should have been replaced"
    x = torch.rand(1, 3, 48, 64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == x.shape
