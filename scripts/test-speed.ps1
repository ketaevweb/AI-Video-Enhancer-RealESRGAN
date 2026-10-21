#requires -version 5.1
<#
=====================================================================
 AI Video Enhancer - realtime speed check for every TensorRT engine
=====================================================================
 Runs each engine through VapourSynth (vspipe) on 240 synthetic
 frames and reports the achieved fps. For smooth playback of 24fps
 movies you want >= 24 fps on the engines your profiles use.
=====================================================================
#>
param([string]$Base = 'C:\AIVideoEnhancer')

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

$EnginesDir = Join-Path $Base 'engines'
$LogsDir    = Join-Path $Base 'logs'
if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }

function Write-Ok($m)  { Write-Host "    [OK] $m" -ForegroundColor Green }
function Write-Wrn($m) { Write-Host "    [!]  $m" -ForegroundColor Yellow }
function Write-Bad($m) { Write-Host "    [X]  $m" -ForegroundColor Red }

# ---- locate vspipe -------------------------------------------------
$vspipe = $null
$cands = @()
try { $cands += (cmd /c 'where vspipe.exe 2>nul') } catch { }
$cands += Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\Scripts\vspipe.exe" -ErrorAction SilentlyContinue |
          Select-Object -ExpandProperty FullName
$cands += 'C:\Program Files\VapourSynth\bin\vspipe.exe'
foreach ($c in $cands) { if ($c -and (Test-Path $c)) { $vspipe = $c; break } }
if (-not $vspipe) { Write-Bad 'vspipe.exe not found - is VapourSynth installed?'; exit 1 }

$engines = Get-ChildItem $EnginesDir -Filter '*.engine' -ErrorAction SilentlyContinue
if (-not $engines) {
    Write-Bad "No engines found in $EnginesDir - run build-engines.ps1 first."
    exit 1
}

Write-Host ''
Write-Host '=====================================================' -ForegroundColor Cyan
Write-Host ' TensorRT engine speed test (240 frames each)'          -ForegroundColor Cyan
Write-Host '=====================================================' -ForegroundColor Cyan

$results = @()
foreach ($eng in $engines) {
    $name = $eng.BaseName
    if ($name -notmatch '_1x3x(\d+)x(\d+)_fp16$') { continue }
    $H = [int]$Matches[1]; $W = [int]$Matches[2]

    $benchVpy = Join-Path $LogsDir 'benchmark.vpy'
    $engineEsc = $eng.FullName.Replace('\', '\\')
@"
import os
import vapoursynth as vs
from vapoursynth import core

base = r"$Base"
names = ('vstrt.dll', 'vs_mlrt.dll')
if not hasattr(core, 'trt'):
    mem = os.path.join(base, 'vsmlrt.path.txt')
    loaded = False
    cands = []
    if os.path.isfile(mem):
        try:
            with open(mem, 'r', encoding='utf-8') as f: cands.append(f.read().strip())
        except Exception: pass
    for n in names:
        cands.append(os.path.join(base, 'vs-mlrt', n))
    for root, dirs, files in os.walk(base):
        hits = [os.path.join(root, n) for n in names if n in files]
        if hits:
            cands.extend(hits)
            break
    for c in cands:
        if c and os.path.isfile(c):
            libdir = os.path.join(os.path.dirname(c), 'vsmlrt-cuda')
            if os.path.isdir(libdir):
                try: os.add_dll_directory(libdir)
                except Exception: pass
                os.environ['PATH'] = libdir + os.pathsep + os.environ.get('PATH', '')
            try:
                core.std.LoadPlugin(c)
            except Exception:
                pass
            if hasattr(core, 'trt'):
                loaded = True
                try:
                    with open(mem, 'w', encoding='utf-8') as f: f.write(c)
                except Exception: pass
                break
    if not loaded:
        raise RuntimeError('vstrt plugin (vstrt.dll / vs_mlrt.dll) not available')

clip = core.std.BlankClip(width=$W, height=$H, format=vs.RGBS, fpsnum=24000, fpsden=1001, length=240)
up = core.trt.Model(clip, engine_path="$engineEsc", num_streams=2)
# Mirror the real .vpy pipelines: YUV420P10 out - mpv's vf_vapoursynth
# accepts YUV output ONLY, RGB fails filter init inside mpv.
core.resize.Bicubic(
    up, format=vs.YUV420P10, matrix_s="709", range_s="limited",
    dither_type="ordered").set_output()
"@ | Set-Content -Path $benchVpy -Encoding UTF8

    $errLog = Join-Path $LogsDir 'benchmark.err.log'
    Write-Host ('  {0}  ({1}x{2} -> x4)...' -f $name, $W, $H)
    cmd /c "`"$vspipe`" --start 0 --end 239 --progress `"$benchVpy`" - >NUL 2>`"$errLog`""

    $fps = 0.0
    if (Test-Path $errLog) {
        $m2 = Select-String -Path $errLog -Pattern '([\d.]+) fps' |
              Select-Object -Last 1
        if ($m2) { $fps = [double]$m2.Matches[0].Groups[1].Value }
    }

    if ($fps -ge 24) {
        Write-Ok ("{0:N1} fps  - real-time for 24fps movies" -f $fps)
        $verdict = 'OK'
    } elseif ($fps -ge 16) {
        Write-Wrn ("{0:N1} fps  - borderline (may drop frames on heavy scenes)" -f $fps)
        $verdict = 'borderline'
    } elseif ($fps -gt 0) {
        Write-Bad ("{0:N1} fps  - too slow for real-time" -f $fps)
        $verdict = 'slow'
    } else {
        Write-Bad 'engine failed to run - see the log:'
        Get-Content $errLog -Tail 10 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
        $verdict = 'failed'
    }
    $results += [pscustomobject]@{ Engine = $name; FPS = $fps; Verdict = $verdict }
    Write-Host ''
}

Write-Host '====================================================='
Write-Host (' {0}  Results' -f $Base)
$results | Format-Table Engine, @{n='fps';e={'{0:N1}' -f $_.FPS}}, Verdict -AutoSize
Write-Host 'Engines marked OK are safe for real-time playback of 24fps movies.'
Write-Host ''
