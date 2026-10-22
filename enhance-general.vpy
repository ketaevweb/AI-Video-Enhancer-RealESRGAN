# -*- coding: utf-8 -*-
# =====================================================================
#  enhance-general.vpy - "natural film" smart profile (QUALITY v8)
#  PRIMARY profile for live-action films on a 2K (2560x1440) monitor.
#
#  WHY v8 EXISTS: PERF v6 made realesr-animevideov3 the primary
#  engine - a model trained on ANIME video. On live action it paints
#  the frame: flat cel-like fills, erased skin/fabric texture and
#  film grain, drawn outlines -> "every movie looks like a cartoon".
#  A compact realtime x4 model fed 640x360 also INVENTS detail, which
#  makes the effect worse. And for clean 720p sources a realtime x4
#  compact model does more harm than good even at its best.
#
#  QUALITY v8 - two paths, chosen by source resolution:
#
#  HD (height >= 600, i.e. 720p/1080p):
#    NO neural network at all. A high-quality Spline36 resize into
#    the 2560x1440 container. 720p -> 2K is only a 2x stretch: real
#    detail beats what a compact realtime model can invent, at zero
#    GPU cost and max fps. This is the honest picture.
#
#  SD (height < 600, DVD-era 480p/576p rips):
#    realesr-general-x4v3 (photo/general compact - NOT the anime
#    model) @ 640x360 engine input -> 2560x1440, then a share of the
#    Spline36-upscaled ORIGINAL is merged back on top (luma full,
#    chroma half). That restores real grain/skin texture over the
#    model output and kills the "AI plastic" look.
#    Tune with environment variable AIVH_DETAIL (default 0.35):
#      0.0 = pure model output (max "detail", more plastic)
#      0.6+ = heavily natural, softer
#
#  HOTFIX v8a (two real bugs found by reproducing the user's crash on a
#  1280x538 movie):
#  1) "Resize error 3074: no path between colorspaces (2/2/2 => 5/2/2)":
#     matrix_s was passed to a SAME-FAMILY YUV->YUV resize. VapourSynth
#     then tries a full colorspace conversion from the frame's
#     unspecified matrix (2) to the requested one (5=BT.470bg) and zimg
#     refuses. Matrix params are ONLY valid when the color family
#     changes (YUV<->RGB); they are now dropped from YUV->YUV hops.
#     (Verified in a local VapourSynth R79 repro: full fixed SD+HD
#     chains pass, including the texture-restore Merge.)
#  2) Scope "720p" rips like 1280x538 were classified as SD (height-only
#     check) and wrongly pushed through the model. HD is now
#     height >= 600 OR width >= 1200. The BT.601/709 matrix guess uses
#     the same rule - a 1280-wide rip is 709-era, not 601.
#
#  Used by mpv:  --vf=vapoursynth=enhance-general.vpy
#  mpv provides the current frame as the "video_in" clip.
# =====================================================================
# AIVH PIPELINE VERSION: QUALITY v8
import os
import vapoursynth as vs
from vapoursynth import core

BASE   = r"C:\AIVideoEnhancer"
ENGINE = os.path.join(BASE, "engines", "realesr-general-x4v3_1x3x360x640_fp16.engine")
IN_W   = 640     # static TensorRT engine input size (SD path)
IN_H   = 360
SCALE  = 4       # general-x4v3 upscales 4x -> 2560x1440
OUT_W  = 2560    # 2K container for the HD bypass path
OUT_H  = 1440

DETAIL = min(max(float(os.environ.get("AIVH_DETAIL", "0.35")), 0.0), 1.0)


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

# mpv's vf_vapoursynth forbids pulling frames during script evaluation
# ("Frame requested during init!" -> black dummy frame), so the color
# matrix is derived from the resolution instead of per-frame props:
#   720p-class sources (height >= 600 OR width >= 1200, which covers
#   scope rips like 1280x538) -> BT.709,  smaller -> BT.601 (470bg)
# This matches virtually every movie rip in existence.
is_hd = (src.height >= 600) or (src.width >= 1200)
mat = "709" if is_hd else "470bg"
rng = "limited"   # YUV handed over by mpv is virtually always limited range

if is_hd:
    # ---------------------------------------------------------------
    # HD path (720p/1080p, incl. scope rips like 1280x538): NO neural
    # network. Honest Spline36 into the 2K container, aspect preserved,
    # output size may be smaller than 2560x1440 for wide aspect ratios
    # - mpv scales to the window. YUV->YUV: NO matrix params allowed
    # (v8a hotfix - they trigger a bogus colorspace conversion and
    # zimg error 3074).
    # ---------------------------------------------------------------
    scale = min(OUT_W / src.width, OUT_H / src.height)
    fw = max(2, int(round(src.width * scale / 2)) * 2)
    fh = max(2, int(round(src.height * scale / 2)) * 2)
    out = core.resize.Spline36(
        src, fw, fh, format=vs.YUV420P10,
        range_s="limited", dither_type="ordered")
else:
    # ---------------------------------------------------------------
    # SD path (480p/576p): general-x4v3 + texture restore.
    # ---------------------------------------------------------------
    if not os.path.isfile(ENGINE):
        raise RuntimeError(
            "TensorRT engine not found:\n  %s\nBuild it with build-engines.ps1" % ENGINE)

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

    # YUV420P10 out (limited range): mpv's vf_vapoursynth accepts YUV
    # output ONLY - a script returning RGB fails filter init ("Unsupported
    # output format") and mpv silently disables the AI. zimg converts the
    # model's float RGB to 10-bit YUV with ordered dither; gpu-next
    # renders it as-is.
    out = core.resize.Bicubic(
        up, format=vs.YUV420P10, matrix_s=mat, range_s="limited",
        dither_type="ordered")

    # Texture restore: merge a share of the Spline36-upscaled ORIGINAL
    # back on top of the model output. Real grain/skin/fabric texture
    # returns, the plastic "AI cartoon" look disappears. Luma gets the
    # full share, chroma half (SD chroma is blotchy 4:2:0 - keep less).
    # YUV->YUV same family: NO matrix_s (v8a hotfix - see header).
    if DETAIL > 0.001:
        orig = core.resize.Spline36(
            src, out.width, out.height, format=vs.YUV420P10,
            range_s="limited", dither_type="ordered")
        out = core.std.Merge(
            out, orig, weight=[DETAIL, DETAIL * 0.5, DETAIL * 0.5])

out.set_output()
