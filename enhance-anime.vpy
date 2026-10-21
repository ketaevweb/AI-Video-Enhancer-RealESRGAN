# -*- coding: utf-8 -*-
# =====================================================================
#  enhance-anime.vpy - realesr-animevideov3 (compact) for mpv
#  Purpose-built anime/animation model. Very fast, temporally stable.
#  Engine input: 960x540, output x4.
# =====================================================================
import os
import vapoursynth as vs
from vapoursynth import core

BASE   = r"C:\AIVideoEnhancer"
ENGINE = os.path.join(BASE, "engines", "realesr-animevideov3_1x3x540x960_fp16.engine")
IN_W   = 960
IN_H   = 540
SCALE  = 4


def _ensure_vstrt():
    if hasattr(core, "trt"):
        return
    cands = []
    mem = os.path.join(BASE, "vsmlrt.path.txt")
    if os.path.isfile(mem):
        try:
            with open(mem, "r", encoding="utf-8") as f:
                cands.append(f.read().strip())
        except Exception:
            pass
    cands.append(os.path.join(BASE, "vs-mlrt", "vs_mlrt.dll"))
    for root, _dirs, files in os.walk(BASE):
        if "vs_mlrt.dll" in files:
            cands.append(os.path.join(root, "vs_mlrt.dll"))
            break
    for c in cands:
        if c and os.path.isfile(c):
            core.std.LoadPlugin(c)
            try:
                with open(mem, "w", encoding="utf-8") as f:
                    f.write(c)
            except Exception:
                pass
            if hasattr(core, "trt"):
                return
    raise RuntimeError(
        "vs_mlrt.dll not found under %s - re-run install.ps1" % BASE)


_ensure_vstrt()
if not hasattr(core, "trt"):
    raise RuntimeError("vstrt (vs-mlrt) plugin did not load")

if not os.path.isfile(ENGINE):
    raise RuntimeError(
        "TensorRT engine not found:\n  %s\nBuild it with build-engines.ps1" % ENGINE)

src = video_in

frame = src.get_frame(0)
props = frame.props


def _pi(key, default):
    try:
        return int(props[key])
    except Exception:
        return default


matrix = _pi("_Matrix", 1)
mat = {1: "709", 5: "470bg", 6: "470bg", 7: "240m", 9: "2020ncl"}.get(matrix, "709")
rng = "full" if _pi("_ColorRange", 1) == 0 else "limited"

if src.format.color_family == vs.RGB:
    rgb = core.resize.Bicubic(src, format=vs.RGBS)
else:
    rgb = core.resize.Bicubic(
        src, format=vs.RGBS, matrix_in_s=mat, range_in_s=rng)

scale = min(IN_W / src.width, IN_H / src.height)
fw = max(2, int(round(src.width * scale / 2)) * 2)
fh = max(2, int(round(src.height * scale / 2)) * 2)

fit = core.resize.Lanczos(rgb, fw, fh)
padded = (fw, fh) != (IN_W, IN_H)
if padded:
    fit = core.std.AddBorders(fit, right=IN_W - fw, bottom=IN_H - fh)

up = core.trt.Model(fit, engine_path=ENGINE)

if padded:
    up = core.std.Crop(up, right=(IN_W - fw) * SCALE, bottom=(IN_H - fh) * SCALE)

out = core.resize.Bicubic(up, format=vs.YUV420P10, matrix_s=mat, range_s=rng)
out.set_output()
