"""VMAF via libvmaf, and the yuv420 writing it needs.

VMAF is an overall-frame metric here. Its foreground variant is computed on
a union-bbox crop and is banned from foreground claims — see
presley.compare.BANNED_FG_KEYS."""

import os
import json
import numpy as np
import subprocess
import tempfile
from typing import Dict, Any, List
from presley.preprocessing import get_reference_frames, get_ufo_masks
_REF_CACHE: Dict[Any, Any] = {}
_MASK_CACHE: Dict[Any, Any] = {}
_DISTS_CACHE: Dict[str, Any] = {}


def calculate_vmaf(ref_yuv: str, dec_video: str, width: int, height: int, framerate: float) -> Dict[str, float]:
    dec_yuv = dec_video + ".yuv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", dec_video, "-pix_fmt", "yuv420p", dec_yuv], check=True)
    
    out_json = dec_video + "_vmaf.json"
    cmd = ["vmaf", "-r", ref_yuv, "-d", dec_yuv, "-w", str(width), "-h", str(height), "-p", "420", "-b", "8", "--json", "-o", out_json]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        with open(out_json, "r") as f:
            data = json.load(f)
        os.remove(dec_yuv)
        os.remove(out_json)
        
        if 'pooled_metrics' in data and 'vmaf' in data['pooled_metrics']:
            v = data['pooled_metrics']['vmaf']
            return {"mean": v.get("mean", 0), "std": v.get("stddev", 0)}
        elif 'frames' in data:
            scores = [f['metrics']['vmaf'] for f in data['frames']]
            return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    except Exception as e:
        print(f"VMAF failed: {e}")
    return {"mean": 0.0, "std": 0.0}
def _write_yuv420(frames: List[np.ndarray], path: str) -> None:
    """Write BGR frames as a raw yuv420p file via an ffmpeg pipe."""
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-i', '-',
         '-pix_fmt', 'yuv420p', '-f', 'rawvideo', path],
        stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
def _vmaf_on_frames(refs: List[np.ndarray], decs: List[np.ndarray], neg: bool = False) -> Dict[str, float]:
    """Run the vmaf CLI on two equal-length BGR frame lists. neg=True uses the
    enhancement-robust vmaf_v0.6.1neg model (returns zeros if unavailable)."""
    h, w = refs[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        ref_yuv, dec_yuv = os.path.join(td, "ref.yuv"), os.path.join(td, "dec.yuv")
        out_json = os.path.join(td, "vmaf.json")
        _write_yuv420(refs, ref_yuv)
        _write_yuv420(decs, dec_yuv)
        cmd = ['vmaf', '-r', ref_yuv, '-d', dec_yuv, '-w', str(w), '-h', str(h),
               '-p', '420', '-b', '8', '--json', '-o', out_json]
        if neg:
            cmd += ['--model', 'version=vmaf_v0.6.1neg']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_json):
            return {"mean": 0.0, "std": 0.0}
        with open(out_json) as f:
            data = json.load(f)
        pooled = data.get('pooled_metrics', {})
        # pooled key is 'vmaf' for the default model; the neg model logs under
        # its own name — take whichever vmaf* key is present
        for k in pooled:
            if k.startswith('vmaf'):
                return {"mean": pooled[k].get("mean", 0.0), "std": pooled[k].get("stddev", 0.0)}
        return {"mean": 0.0, "std": 0.0}
