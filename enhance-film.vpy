# -*- coding: utf-8 -*-
# =====================================================================
#  enhance-film.vpy - Real-ESRGAN (RealESRGAN_x4plus, RRDB) for mpv
#  Best-detail live-action profile. Engine input: 960x540, output x4.
#
#  Used by mpv:  --vf=vapoursynth=C\:/AIVideoEnhancer/enhance-film.vpy
#  mpv provides the current frame as the "video_in" clip.
# =====================================================================
import os
import vapoursynth as vs
from vapoursynth import core

BASE   = r"C:\AIVideoEnhancer"
ENGINE = os.path.join(BASE, "engines", "RealESRGAN_x4plus_1x3x540x960_fp16.engine")
IN_W   = 960     # static TensorRT engine input size
IN_H   = 540
SCALE  = 4       # RealESRGAN_x4plus upscales 4x


# ---------------------------------------------------------------------
# Load the vs-mlrt (vstrt) plugin if it is not already auto-loaded.
# FIX v3: vs-mlrt v15 renamed the plugin to vstrt.dll (vs_mlrt.dll is
# the legacy name). The TensorRT runtime DLLs live in the vsmlrt-cuda
# subfolder, so it is registered via os.add_dll_directory() + PATH
# before LoadPlugin - no PATH edit and no DLL copying required.
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

# ---------------------------------------------------------------------
# 1. Color metadata: mpv's vf_vapoursynth forbids pulling frames during
#    script evaluation ("Frame requested during init!" -> black dummy
#    frame), so the matrix comes from the resolution instead of props:
#    < 600 lines (480p/576p) -> BT.601,   720p and above -> BT.709
# ---------------------------------------------------------------------
mat = "709" if src.height >= 600 else "470bg"
rng = "limited"   # YUV handed over by mpv is virtually always limited range

# ---------------------------------------------------------------------
# 2. To planar float RGB
# ---------------------------------------------------------------------
if src.format.color_family == vs.RGB:
    rgb = core.resize.Bicubic(src, format=vs.RGBS)
else:
    rgb = core.resize.Bicubic(
        src, format=vs.RGBS, matrix_in_s=mat, range_in_s=rng)

# ---------------------------------------------------------------------
# 3. Fit into the static engine input, preserving aspect ratio.
#    Content is resized to (fw, fh) and padded to (IN_W, IN_H);
#    the padding is cropped away after inference. Nothing is distorted.
# ---------------------------------------------------------------------
scale = min(IN_W / src.width, IN_H / src.height)
fw = max(2, int(round(src.width * scale / 2)) * 2)
fh = max(2, int(round(src.height * scale / 2)) * 2)

fit = core.resize.Lanczos(rgb, fw, fh)
padded = (fw, fh) != (IN_W, IN_H)
if padded:
    fit = core.std.AddBorders(fit, right=IN_W - fw, bottom=IN_H - fh)

# ---------------------------------------------------------------------
# 4. "Honest" Real-ESRGAN inference on the GPU (TensorRT, fp16)
# ---------------------------------------------------------------------
up = core.trt.Model(fit, engine_path=ENGINE, num_streams=2)

if padded:
    up = core.std.Crop(up, right=(IN_W - fw) * SCALE, bottom=(IN_H - fh) * SCALE)

# ---------------------------------------------------------------------
# 5. Output YUV420P10 (limited range). mpv's vf_vapoursynth accepts YUV
#    output ONLY - a script returning RGB fails filter init ("Unsupported
#    output format") and mpv silently disables the AI. zimg converts the
#    model's float RGB to 10-bit YUV with ordered dithering, gpu-next
#    then renders it directly on the GPU.
# ---------------------------------------------------------------------
out = core.resize.Bicubic(
    up, format=vs.YUV420P10, matrix_s=mat, range_s="limited",
    dither_type="ordered")
out.set_output()
