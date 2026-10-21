# -*- coding: utf-8 -*-
# =====================================================================
#  enhance-general.vpy - realesr-animevideov3 (compact, 621K params)
#  PRIMARY realtime profile for a 2K (2560x1440) monitor.
#
#  PERF v6: measured on RTX 3080 / driver 617.42, 720p source:
#    general-x4v3 @ 960x540 in  ->   7 fps in mpv   (unusable)
#    animevideov3 @ 960x540 in  ->  23 fps in mpv   (watchable)
#  Compute scales with the model OUTPUT pixel count, so the engine now
#  takes 640x360 input and produces 2560x1440 directly:
#    animevideov3 @ 640x360 in  -> ~50 fps in mpv   (large headroom,
#    no more A/V desync, GPU no longer pegged at the frame budget)
#
#  The model was trained for low-quality video streams, so it also
#  DENOISES compressed/old rips by itself - the separate wdn blend of
#  the old x4v3 pipeline is gone (it doubled the frame cost).
#  Max-detail live-action alternative: enhance-film.vpy.
# =====================================================================
import os
import vapoursynth as vs
from vapoursynth import core

BASE   = r"C:\AIVideoEnhancer"
ENGINE = os.path.join(BASE, "engines", "realesr-animevideov3_1x3x360x640_fp16.engine")
IN_W   = 640     # static TensorRT engine input size
IN_H   = 360
SCALE  = 4       # animevideov3 upscales 4x -> 2560x1440


# --- resilient plugin loader (FIX v3: vstrt.dll in v15+) -------------
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

src = video_in

# mpv's vf_vapoursynth forbids pulling frames during script evaluation
# ("Frame requested during init!" -> black dummy frame), so the color
# matrix is derived from the resolution instead of per-frame props:
#   < 600 lines (480p/576p DVD-era rips) -> BT.601,   720p+ -> BT.709
# This matches virtually every movie rip in existence.
mat = "709" if src.height >= 600 else "470bg"
rng = "limited"   # YUV handed over by mpv is virtually always limited range

if src.format.color_family == vs.RGB:
    rgb = core.resize.Bicubic(src, format=vs.RGBS)
else:
    rgb = core.resize.Bicubic(
        src, format=vs.RGBS, matrix_in_s=mat, range_in_s=rng)

# Fit into the static engine input, preserving aspect ratio.
# Content is resized to (fw, fh) and padded to (IN_W, IN_H); the
# padding is cropped away after inference. Nothing is distorted.
scale = min(IN_W / src.width, IN_H / src.height)
fw = max(2, int(round(src.width * scale / 2)) * 2)
fh = max(2, int(round(src.height * scale / 2)) * 2)

fit = core.resize.Lanczos(rgb, fw, fh)
padded = (fw, fh) != (IN_W, IN_H)
if padded:
    fit = core.std.AddBorders(fit, right=IN_W - fw, bottom=IN_H - fh)

up = core.trt.Model(fit, engine_path=ENGINE, num_streams=2)

if padded:
    up = core.std.Crop(up, right=(IN_W - fw) * SCALE, bottom=(IN_H - fh) * SCALE)

# YUV420P10 out (limited range): mpv's vf_vapoursynth accepts YUV output
# ONLY - a script returning RGB fails filter init ("Unsupported output
# format") and mpv silently disables the AI. zimg converts the model's
# float RGB to 10-bit YUV with ordered dither; gpu-next renders it as-is.
out = core.resize.Bicubic(
    up, format=vs.YUV420P10, matrix_s=mat, range_s="limited",
    dither_type="ordered")
out.set_output()
