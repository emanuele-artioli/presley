# Wave 1 — cheap falsifiers

Each item is <=1 day and is designed to **kill its own workstream** cheaply,
before any multi-day build. Bounds are written *before* the number is read (host
rule: bound before believing); a result outside its stated range is an alarm to
investigate, not a finding to report.

Probe videos come from `tools/select_probe_videos.py`: **camel, motorbike,
drift-straight, dancing** (cluster medoids over `scripts/audit_videos.py`
attributes). `bear` is deliberately excluded -- it never separates from camel at
any k from 2 to 8 (0.92 sd apart vs a 4.15 sd median pairwise distance), so
running both buys nothing.

JND thresholds (`src/presley/compare.py`): PSNR 0.5 dB, SSIM 0.05, LPIPS 0.05,
DISTS 0.05, VMAF 6.0. **Deltas within JND are "no perceptible difference" --
never a trend, never a win.**

| ID | Goal | Test | Status |
|---|---|---|---|
| F1 | 1 | All-intra leave-one-SB-out bit map vs EVCA | todo |
| F2 | 1 | 64x64-snapped vs scattered 16x16 selection | todo |
| **F3** | **2** | **`--tune 0` (VQ) vs the PSNR default** | **DONE — confound confirmed, effect sub-JND** |
| F4 | 2 | `--film-grain` on/off, selective | todo |
| F5 | 2 | Transform-aligned AC truncation vs Gaussian blur | todo |
| F6 | 3 | Encoder-side FG gate: does restoring FG clear JND? | todo |
| F7 | 2 | Chroma-first degradation probe | todo |

---

## F3 — SVT-AV1 `tune`: is the PSNR default a confound?

**Question.** `encode_video_svtav1_qp` passed `rc=0:q={qp}` with no `tune`.
SVT-AV1 offers 0=VQ, 1=PSNR, 2=SSIM. If the default is PSNR, then every
perceptual claim in the paper was measured under an encoder optimizing an
objective the paper does not claim -- a confound under all of it.

**Bounds, stated before measuring.** If unset is bitwise-identical to `tune=1`,
the confound is real. For VQ vs PSNR at matched rate expect LPIPS to improve
0.005-0.03, PSNR to drop 0.2-1.0 dB, bitrate to move <=10%. Alarm if LPIPS moves
>0.1 or bitrate >30% -- too large for an RDO tuning flag.

**Method.** camel + all four probe medoids, 640x360, preset 8, fixed QP.
Baseline `tune=1 q=43`; VQ arm re-encoded at the QP whose byte size lands closest
to it, so the comparison is rate-matched rather than QP-matched (VQ is +6.8% bits
at equal QP, so a same-QP comparison would flatter it). Decoded with ffmpeg --
OpenCV cannot decode AV1 in this environment and returns empty frame lists,
which silently produces NaN metrics.

**Result 1 — the default is PSNR.** With `tune` omitted, SVT-AV1 logs
`tune : PSNR` and emits a file **bitwise identical** to `tune=1` (md5
`4e4625cd50efec6e`, 298158 B). At QP 43 on camel: VQ +6.8% bits, SSIM-tune
-9.4%. Confound confirmed.

**Result 2 — but the effect is sub-JND everywhere.** Rate-matched, VQ vs PSNR:

| video | dPSNR | dLPIPS | dDISTS | dBits |
|---|---|---|---|---|
| camel | -0.14 | -0.0172 | -0.0148 | -0.2% |
| motorbike | -0.21 | -0.0305 | -0.0105 | +1.2% |
| drift-straight | -0.36 | -0.0173 | -0.0106 | -2.5% |
| dancing | -0.11 | -0.0270 | -0.0097 | +2.4% |
| elephant | -0.21 | -0.0150 | -0.0107 | -1.5% |
| **mean** | | **-0.0214** | **-0.0112** | |

(Negative = VQ better on LPIPS/DISTS, worse on PSNR.)

**Verdict.** Direction is consistent 5/5, so it is not video-determined, and it
is exactly what theory predicts: a perceptually-tuned RDO trades a little PSNR
for a little perceptual quality. But the largest LPIPS delta (0.0305) is well
under the 0.05 JND, DISTS never exceeds 0.0148, and the PSNR cost stays inside
its own 0.5 dB JND.

So: **the confound is real but benign.** Past restorer comparisons are not
invalidated by having run PSNR-tuned, because switching the objective does not
move any metric perceptibly. This closes a risk rather than opening one, and VQ
must **not** be reported as a free win -- that would be exactly the
"imperceptible delta dressed up" failure the hard rules forbid.

**Landed.** `tune` is now an optional `codec_params` key on all three SVT-AV1
callers (baselines, elvis, presley_ai), defaulting to omitted so output stays
bit-exact and all 694 existing result hashes remain valid. The point is
reproducibility -- the paper should be able to state the tuning it used rather
than inherit an invisible default.
