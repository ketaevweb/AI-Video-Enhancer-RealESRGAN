# -*- coding: utf-8 -*-
# =====================================================================
#  transcode-source.vpy - OFFLINE source wrapper for batch transcoding
#  AIVH TRANSCODE v1
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
#  - The source is opened with ffms2 (bundled with VapourSynth) or
#    L-SMASH Works as fallback. The first run creates an index next
#    to the file (.ffindex / .lwi) - for a big MKV this takes a
#    minute or two ONCE.
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


def _open_source(path):
    # ffms2 ships with VapourSynth; L-SMASH Works is the fallback.
    if hasattr(core, "ffms2"):
        return core.ffms2.Source(path)
    if hasattr(core, "lsmas"):
        return core.lsmas.LWLibavSource(path)
    raise RuntimeError(
        "Neither ffms2 nor lsmas is available - the VapourSynth install "
        "is incomplete. Re-run install.ps1 (R79 ships ffms2).")


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
