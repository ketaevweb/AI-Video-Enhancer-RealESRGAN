# -*- coding: utf-8 -*-
# =====================================================================
#  enhance-film-hd.vpy - RealESRGAN_x2plus (RRDB) for 720p sources
#  MAX-QUALITY path for 720p: direct 720p -> 1440p (x2), no downscale
#  to 540p, so real detail is preserved.
#
#  OFFLINE (TRANSCODE v2.3): scripts\transcode-runner.ps1 pads the
#  source up to the next multiple of 16 and passes a SHAPE-MATCHED
#  x2plus engine via AIVH_OFFLINE_ENGINE (+ AIVH_OFFLINE_IN_W/H).
#  Non-720p movies then get a true native-resolution 2x upscale
#  (only zero-impact border padding) instead of being squished into
#  the 720x1280 engine, which threw away ~25% of a 970p rip's detail.
#
#  If the x2plus engine has not been built yet, this script
#  automatically falls back to the 540p x4plus engine, so mpv profiles
#  can always reference it safely.
#  QUALITY v8 note: unchanged - this stays the OFFLINE max-quality
#  path (RRDB is far too heavy for realtime viewing).
# AIVH PIPELINE VERSION: QUALITY v9
# =====================================================================
import os
import vapoursynth as vs
from vapoursynth import core

BASE = r"C:\AIVideoEnhancer"

ENGINE_HD = os.path.join(BASE, "engines", "RealESRGAN_x2plus_1x3x720x1280_fp16.engine")
ENGINE_SD = os.path.join(BASE, "engines", "RealESRGAN_x4plus_1x3x540x960_fp16.engine")

IN_W = 1280
IN_H = 720
SCALE = 2


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

src = video_in

# --- engine selection with automatic fallback ------------------------
# (1) OFFLINE shape-matched engine (transcode-runner.ps1 v2.3): the
#     engine input must fully CONTAIN the source frame - the rest is
#     pure border padding, never a resample. Needs src, hence this
#     block sits after 'src = video_in'.
offline_native = False
_off_engine = os.environ.get("AIVH_OFFLINE_ENGINE", "")
try:
    _off_w = int(os.environ.get("AIVH_OFFLINE_IN_W", "") or 0)
    _off_h = int(os.environ.get("AIVH_OFFLINE_IN_H", "") or 0)
except ValueError:
    _off_w = _off_h = 0
if (_off_engine and os.path.isfile(_off_engine) and _off_w > 0 and _off_h > 0
        and _off_w >= src.width and _off_h >= src.height):
    ENGINE = _off_engine
    IN_W, IN_H, SCALE = _off_w, _off_h, 2
    offline_native = True
elif os.path.isfile(ENGINE_HD):
    ENGINE = ENGINE_HD
    IN_W, IN_H, SCALE = 1280, 720, 2
else:
    ENGINE = ENGINE_SD
    IN_W, IN_H, SCALE = 960, 540, 4
    try:
        core.log_message(
            vs.logLevel.warning,
            "x2plus engine not built yet - falling back to the 540p x4plus engine. "
            "Run build-engines.ps1 without -Minimal to enable the HD path.")
    except Exception:
        pass

if not os.path.isfile(ENGINE):
    raise RuntimeError(
        "TensorRT engine not found:\n  %s\nBuild engines with build-engines.ps1" % ENGINE)

# --- color metadata ---------------------------------------------------
# mpv's vf_vapoursynth forbids pulling frames during script evaluation
# ("Frame requested during init!" -> black dummy frame), so the color
# matrix is derived from the resolution instead of per-frame props:
#   < 600 lines (480p/576p DVD-era rips) -> BT.601,   720p+ -> BT.709
mat = "709" if (src.height >= 600 or src.width >= 1200) else "470bg"
rng = "limited"   # YUV handed over by mpv is virtually always limited range

if src.format.color_family == vs.RGB:
    rgb = core.resize.Bicubic(src, format=vs.RGBS)
else:
    rgb = core.resize.Bicubic(
        src, format=vs.RGBS, matrix_in_s=mat, range_in_s=rng)

# --- fit + pad (aspect-ratio safe), infer, crop -----------------------
if offline_native:
    # native-res mode: NO resample at all, only zero-impact padding up
    # to the engine's input shape (both dims multiples of 16)
    fw, fh = src.width, src.height
    fit = rgb
else:
    scale = min(IN_W / src.width, IN_H / src.height)
    fw = max(2, int(round(src.width * scale / 2)) * 2)
    fh = max(2, int(round(src.height * scale / 2)) * 2)
    # identical dims -> skip the resize entirely (it used to be an
    # unnecessary full-frame Lanczos copy even for exact 720x1280)
    fit = rgb if (fw, fh) == (src.width, src.height) else core.resize.Lanczos(rgb, fw, fh)
padded = (fw, fh) != (IN_W, IN_H)
if padded:
    fit = core.std.AddBorders(fit, right=IN_W - fw, bottom=IN_H - fh)

# RRDB activation memory grows linearly with the pixel count: two
# overlapping streams fit a 720x1280 input on a 10 GB card, but a
# 976x1280 shape-matched engine (~1.36x the pixels) runs single-stream.
streams = 2 if (IN_W * IN_H <= 1280 * 720) else 1
up = core.trt.Model(fit, engine_path=ENGINE, num_streams=streams)

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
