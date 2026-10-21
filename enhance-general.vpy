# -*- coding: utf-8 -*-
# =====================================================================
#  enhance-general.vpy - realesr-general-x4v3 (+ optional wdn denoise)
#  PRIMARY realtime profile for live-action films.
#
#  Default: ONLY the clean model runs - the fastest path (~30 fps class
#  on RTX 3080). The optional "wdn" denoise blend removes heavy noise
#  from old DVD/VHS rips but runs a SECOND model per frame, halving the
#  realtime speed. Enable it via:
#    - watch-AI-general-denoise.bat launcher, or
#    - environment variable: set AIVH_DENOISE=0.5 before starting mpv
#
#  DENOISE: 0.0 = pure sharpening (no denoise, fast) ... 1.0 = max denoise
#  0.5 is a good starting point for DVD-quality sources.
# =====================================================================
import os
import vapoursynth as vs
from vapoursynth import core

BASE = r"C:\AIVideoEnhancer"
ENGINE_CLEAN = os.path.join(BASE, "engines", "realesr-general-x4v3_1x3x540x960_fp16.engine")
ENGINE_WDN   = os.path.join(BASE, "engines", "realesr-general-wdn-x4v3_1x3x540x960_fp16.engine")
DENOISE = float(os.environ.get("AIVH_DENOISE", "0"))   # 0 = fastest single-model path; try 0.5 for noisy rips
IN_W    = 960
IN_H    = 540
SCALE   = 4


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

for eng in (ENGINE_CLEAN, ENGINE_WDN):
    if not os.path.isfile(eng):
        raise RuntimeError(
            "TensorRT engine not found:\n  %s\nBuild it with build-engines.ps1" % eng)

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

scale = min(IN_W / src.width, IN_H / src.height)
fw = max(2, int(round(src.width * scale / 2)) * 2)
fh = max(2, int(round(src.height * scale / 2)) * 2)

fit = core.resize.Lanczos(rgb, fw, fh)
padded = (fw, fh) != (IN_W, IN_H)
if padded:
    fit = core.std.AddBorders(fit, right=IN_W - fw, bottom=IN_H - fh)

clean = core.trt.Model(fit, engine_path=ENGINE_CLEAN, num_streams=2)
if DENOISE > 0:
    wdn   = core.trt.Model(fit, engine_path=ENGINE_WDN, num_streams=2)
    up    = core.std.Merge(clean, wdn, weight=[DENOISE])
else:
    up    = clean   # single-model fast path: double the realtime headroom

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
