#!/usr/bin/env python3
"""W1g: race the corrected selection objective against the incumbent.

Two arms, identical in everything except which score ranks the blocks:

  control    selection_rule 'removability' -- the existing score, a bits-cost
             proxy and nothing else (Eq. importance)
  corrected  selection_rule 'restorability' -- that score divided by predicted
             post-restoration damage, the correction the M1 diagnosis implies

The clustering blur, the hard foreground exclusion, the budget, the encoder,
the QP rungs and the restorer are all held fixed, so the arms differ in WHICH
blocks are degraded and in nothing else. Verified on bear: both degrade exactly
73800 blocks.

**Six videos, all with a held-out damage-model fold.** `presley.damagemodel.load`
raises rather than scoring a clip with a model that trained on it, so the arm
cannot silently leak; this list is the set that survives that constraint and
overlaps the corpus the article already reports. `train` is deliberately absent
despite being in the resolution ladder -- it has no fold, and adding one would
mean refitting on data that includes it.

Five of the six control cells already exist as the 360p rung of the resolution
ladder (same recipe, same hashes), so only their corrected twins are new.

Pre-registered bounds, stated before the numbers are read
(PLAN_SUBMISSION_PREP.md 3.3): BD-rate of corrected against control in
-20%..+25%. The sign is genuinely not predicted. What IS known in advance is
that the corrected rule starts down on the rate axis -- measured end to end on
bear at QP 50 it moves the transport +3.6% -- so it has to earn that back
through restoration quality rather than through bits.

Usage:
    python scripts/build_selection_race.py -o config/w1g_selection_race.yaml
"""
from __future__ import annotations

import argparse
import sys

import yaml

# Every one of these has a leave-one-out fold in config/damage_predictor.json.
VIDEOS = ("bear", "camel", "dog", "pigs", "tennis", "india")

QPS = (43, 50, 55, 60)
WIDTH, HEIGHT, BLOCK = 640, 360, 8


def entry(video: str, qp: int, rule: str) -> dict:
    cfg = {
        "component": "presley_ai",
        "video": video,
        "width": WIDTH,
        "height": HEIGHT,
        "block_size": BLOCK,
        "codec": "svtav1",
        "codec_params": {"preset": "8", "qp": qp},
        "alpha": 0.5,
        "beta": 0.5,
        "shrink_amount": 0.25,
        "fg_protect": True,
        "composite_output": True,
        "degradation": "downsample",
        "restorer": "realesrgan",
        "restorer_params": {},
        "target_bitrate": 0,
    }
    if rule == "restorability":
        # The control arm carries NO selection_rule key, so its hash is
        # identical to the runs the resolution ladder already produced. Adding
        # the key only to the corrected arm is what makes five of the six
        # control cells free.
        cfg["selection_rule"] = "restorability"
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    entries = [entry(v, qp, rule)
               for v in VIDEOS for qp in QPS
               for rule in ("removability", "restorability")]
    with open(args.out, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"experiments": entries}, fh, sort_keys=True,
                       default_flow_style=False)
    print(f"{args.out}: {len(entries)} entries "
          f"({len(VIDEOS)} videos x {len(QPS)} QPs x 2 arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
