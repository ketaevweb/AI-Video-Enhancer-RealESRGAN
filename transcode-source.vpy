# -*- coding: utf-8 -*-
# =====================================================================
#  transcode-source.vpy - OFFLINE source wrapper for batch transcoding
#  AIVH TRANSCODE v2.3
# =====================================================================
#  The realtime profiles (enhance-*.vpy) get their frames from mpv as
#  the "video_in" clip. Offline there is no mpv - this wrapper opens
#  the movie from DISK, hands it to a profile script as video_in
#  (with the same frame props mpv would provide), and lets the
#  profile do its normal work. The profile's set_output() is what
#  vspipe streams out.
#
#  Driven by scripts\transcode-runner.ps1 (transcode-AI.bat), i.e.:
#
#    vspipe -c y4m transcode-source.vpy - | ffmpeg -f y4m -i - ...
#
#  Environment variables (set by transcode-AI.bat):
#    AIVH_SRC               full path of the movie to transcode  (required)
#    AIVH                   C:\AIVideoEnhancer (profile location)
#    AIVH_TRANSCODE_PROFILE enhance-film-hd.vpy | enhance-film.vpy |
#                           enhance-general.vpy | enhance-anime.vpy
#
#  Notes:
#  - The source is opened with ffms2 or L-SMASH Works as fallback.
#    v2.3 reality check: a pip-installed VapourSynth (which is what
#    install.ps1 sets up) ships NO source filters at all - the old
#    comment "R79 ships ffms2" was only ever true for the standalone
#    installer. install.ps1 therefore downloads the official FFMS2
#    plugin into C:\AIVideoEnhancer\plugins\ffms2.dll, and this
#    wrapper loads it explicitly (same pattern as the vstrt plugin).
#    If opening still fails, the raised error now lists EVERY attempt
#    with the actual exception instead of a canned misleading text.
#  - The first run creates an index next to the file (.ffindex /
#    .lwi) - for a big MKV this takes a minute or two ONCE.
#  - _Matrix/_Primaries/_Transfer are forced to 2 (unspecified) so
#    the profiles use exactly the same resolution-based BT.601/709
#    guess they use under mpv - identical colors in realtime and in
#    the transcoded file.
#  - Frame rate comes from the container. Virtually every movie rip
#    is CFR; VFR sources are treated as CFR (same as the realtime
#    path in mpv).
# =====================================================================
import os
import vapoursynth as vs
from vapoursynth import core

BASE = os.environ.get("AIVH", r"C:\AIVideoEnhancer")

SRC = os.environ.get("AIVH_SRC", "")
if not SRC:
    raise RuntimeError(
        "AIVH_SRC is not set. Do not run this script directly - "
        "drag the movie onto transcode-AI.bat (scripts\\transcode-runner.ps1)")
if not os.path.isfile(SRC):
    raise RuntimeError("AIVH_SRC does not point to an existing file: %s" % SRC)


def _try_source(ns, func, path, errs):
    # returns the clip, or None if the namespace is absent / open failed
    if not hasattr(core, ns):
        return None
    try:
        clip = getattr(getattr(core, ns), func)(path)
        return clip if clip is not None else None
    except Exception as e:
        errs.append("%s.%s: %r" % (ns, func, e))
        return None


def _open_source(path):
    # (1) already-autoloaded plugins (standalone-installer layout)
    errs = []
    clip = _try_source("ffms2", "Source", path, errs)
    if clip is not None:
        return clip
    clip = _try_source("lsmas", "LWLibavSource", path, errs)
    if clip is not None:
        return clip
    # (2) explicit LoadPlugin from the install dir - pip-installed
    #     VapourSynth ships no source filters, install.ps1 puts the
    #     official FFMS2 plugin here (same pattern as vstrt)
    pdir = os.path.join(BASE, "plugins")
    for dll, ns, func in (("ffms2.dll", "ffms2", "Source"),
                          ("lsmas.dll", "lsmas", "LWLibavSource")):
        if hasattr(core, ns):   # autoloaded above and failed on this file
            continue
        dll_path = os.path.join(pdir, dll)
        if not os.path.isfile(dll_path):
            continue
        try:
            core.std.LoadPlugin(dll_path)
        except Exception as e:
            errs.append("LoadPlugin(%s): %r" % (dll_path, e))
            continue
        clip = _try_source(ns, func, path, errs)
        if clip is not None:
            return clip
    # (3) nothing worked - report exactly what was tried
    msg = ["Neither ffms2 nor lsmas could open this movie."]
    if errs:
        msg.append("What was tried:")
        msg.extend("  - " + e for e in errs)
    else:
        msg.append("No source filter is installed: pip-installed VapourSynth")
        msg.append("ships none and %s is missing." % os.path.join(pdir, "ffms2.dll"))
    msg.append("Fix: re-run install.ps1 (downloads the ffms2 source reader),")
    msg.append("or start the job again - transcode-AI.bat fetches it itself.")
    raise RuntimeError("\n".join(msg))


video_in = _open_source(SRC)

# mpv hands frames over with unspecified color metadata on virtually
# every rip; forcing the same props here keeps the profile's matrix
# guess (and therefore the colors) identical to realtime playback.
video_in = core.std.SetFrameProps(video_in, _Matrix=2, _Primaries=2, _Transfer=2)

PROFILE = os.environ.get("AIVH_TRANSCODE_PROFILE", "enhance-film-hd.vpy")
_profile_path = PROFILE if os.path.isabs(PROFILE) else os.path.join(BASE, PROFILE)
if not os.path.isfile(_profile_path):
    raise RuntimeError("profile script not found: %s" % _profile_path)

with open(_profile_path, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _profile_path, "exec"), globals())
