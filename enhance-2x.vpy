# -*- coding: utf-8 -*-
# =====================================================================
#  enhance-2x.vpy - SPEED profile for 2x playback (QUALITY v10)
#  NEW: for people who watch films at double speed. At --speed=2.0
#  mpv consumes frames TWICE as fast (a 24 fps film demands 48 fps),
#  so the quality profiles (general ~22 fps, film ~10-15 fps) run out
#  of budget and the picture stutters. This profile flips the trade:
#  the LIGHTEST engine in the set renders every frame with headroom.
#
#  Model: realesr-animevideov3 @ 640x360 engine input -> 2560x1440.
#  It is the smallest network of the project (1.8 MB fp16 engine,
#  32-feature compact net vs 64-feature x4v3) - measured 2x+ faster
#  than enhance-film.vpy at the same output size. ~50-70 fps on an
#  RTX 3080 -> 2x of 24/25 fps films is stable, 2x of 30 fps mostly
#  stable (mpv --framedrop=decoder+vo absorbs the spikes).
#
#  The anime-model look on live action is tamed the same way the
#  general profile does it: a share of the Spline36-upscaled ORIGINAL
#  is merged back on top (luma full, chroma half), which returns real
#  grain/skin texture over the painted output. Tune with the
#  AIVH_DETAIL environment variable (default 0.35), same semantics as
#  enhance-general.vpy. AIVH_DETAIL=0 gives the pure model output.
#
#  Used by: watch-AI-2x.bat and the Ctrl+s hotkey in input.conf.
#  Set the speed inside mpv: Ctrl+2 = 2.0x, Ctrl+1 = 1.0x (or the
#  classic `}` / `[` / `]` keys).
# =====================================================================
# AIVH PIPELINE VERSION: QUALITY v10
import os
import vapoursynth as vs
from vapoursynth import core

BASE   = r"C:\AIVideoEnhancer"
ENGINE = os.path.join(BASE, "engines", "realesr-animevideov3_1x3x360x640_fp16.engine")
IN_W   = 640     # static TensorRT engine input size
IN_H   = 360
SCALE  = 4       # animevideov3 upscales 4x -> 2560x1440

DETAIL = min(max(float(os.environ.get("AIVH_DETAIL", "0.35")), 0.0), 1.0)


# ---------------------------------------------------------------------
# Load the vs-mlrt (vstrt) plugin if it is not already auto-loaded.
# Same loader as the other profiles (vstrt.dll in v15+, vsmlrt-cuda
# runtime DLLs registered before LoadPlugin).
# ---------------------------------------------------------------------
def _ensure_vstrt():
    if hasattr(core, "trt"):
        return
    names = ("vstrt.dll", "vs_mlrt.dll")   # v15+ name first, legacy fallback
    cands = []
    mem = os.path.join(BASE, "vsmlrt.path.txt")
    if os.path.isfile(mem):
        try:
            with open(mem, "r", encoding="utf-8") as f:
                cands.append(f.read().strip())
        except Exception:
            pass
    for n in names:
        cands.append(os.path.join(BASE, "vs-mlrt", n))
    for root, _dirs, files in os.walk(BASE):
        hits = [os.path.join(root, n) for n in names if n in files]
        if hits:
            cands.extend(hits)
            break
    last_err = None
    for c in cands:
        if c and os.path.isfile(c):
            libdir = os.path.join(os.path.dirname(c), "vsmlrt-cuda")
            if os.path.isdir(libdir):
                try:
                    os.add_dll_directory(libdir)
                except Exception:
                    pass
                os.environ["PATH"] = libdir + os.pathsep + os.environ.get("PATH", "")
            try:
                core.std.LoadPlugin(c)
            except Exception as e:
                last_err = e
                continue
            try:
                with open(mem, "w", encoding="utf-8") as f:
                    f.write(c)
            except Exception:
                pass
            if hasattr(core, "trt"):
                return
    raise RuntimeError(
        "vstrt plugin (vstrt.dll / vs_mlrt.dll) not found or failed to load "
        "under %s%s - re-run install.ps1"
        % (BASE, (": %r" % last_err) if last_err else ""))


_ensure_vstrt()
if not hasattr(core, "trt"):
    raise RuntimeError("vstrt (vs-mlrt) plugin did not load")

if not os.path.isfile(ENGINE):
    raise RuntimeError(
        "TensorRT engine not found:\n  %s\nBuild it with build-engines.ps1" % ENGINE)


src = video_in  # provided by mpv

# mpv's vf_vapoursynth forbids pulling frames during script evaluation,
# so the matrix comes from the resolution:
#   < 600 lines and < 1200 wide (480p/576p) -> BT.601, else BT.709
mat = "709" if (src.height >= 600 or src.width >= 1200) else "470bg"
rng = "limited"   # YUV handed over by mpv is virtually always limited range

# ---------------------------------------------------------------------
# To planar float RGB (the engine consumes RGBS)
# ---------------------------------------------------------------------
if src.format.color_family == vs.RGB:
    rgb = core.resize.Bicubic(src, format=vs.RGBS)
else:
    rgb = core.resize.Bicubic(
        src, format=vs.RGBS, matrix_in_s=mat, range_in_s=rng)

# ---------------------------------------------------------------------
# Fit into the static engine input, preserving aspect ratio (pad/crop,
# nothing is distorted - same scheme as every other profile here).
# ---------------------------------------------------------------------
scale = min(IN_W / src.width, IN_H / src.height)
fw = max(2, int(round(src.width * scale / 2)) * 2)
fh = max(2, int(round(src.height * scale / 2)) * 2)

fit = core.resize.Lanczos(rgb, fw, fh)
padded = (fw, fh) != (IN_W, IN_H)
if padded:
    fit = core.std.AddBorders(fit, right=IN_W - fw, bottom=IN_H - fh)

# ---------------------------------------------------------------------
# Inference (TensorRT, fp16). num_streams=2 overlaps the copy and the
# compute of adjacent frames - the default for the compact engines.
# ---------------------------------------------------------------------
up = core.trt.Model(fit, engine_path=ENGINE, num_streams=2)

if padded:
    up = core.std.Crop(up, right=(IN_W - fw) * SCALE, bottom=(IN_H - fh) * SCALE)

# ---------------------------------------------------------------------
# Output YUV420P10 (limited range). mpv's vf_vapoursynth accepts YUV
# output ONLY - a script returning RGB fails filter init and mpv
# silently disables the AI.
# ---------------------------------------------------------------------
out = core.resize.Bicubic(
    up, format=vs.YUV420P10, matrix_s=mat, range_s="limited",
    dither_type="ordered")

# Texture restore (same as enhance-general.vpy): merge a share of the
# Spline36-upscaled ORIGINAL back on top. Luma full share, chroma half.
# Costs one zimg resize + a std.Merge per frame (microseconds next to
# the inference) - keeps the natural film texture at 2x speed.
# YUV->YUV same family: NO matrix_s (v8a hotfix).
if DETAIL > 0.001:
    orig = core.resize.Spline36(
        src, out.width, out.height, format=vs.YUV420P10,
        range_s="limited", dither_type="ordered")
    out = core.std.Merge(
        out, orig, weight=[DETAIL, DETAIL * 0.5, DETAIL * 0.5])

out.set_output()
