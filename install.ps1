#requires -version 5.1
<#
=====================================================================
 AI Video Enhancer installer (Real-ESRGAN realtime via VapourSynth)
=====================================================================
 Pipeline:  mpv  ->  VapourSynth (.vpy)  ->  vs-mlrt/vstrt (TensorRT)
            ->  "honest" Real-ESRGAN inference on your NVIDIA GPU
            ->  4K frame back to mpv, rendered on your 2K monitor.

 Idempotent: safe to re-run any time. Finished steps are skipped.

 FIX v2: all native tools (nvidia-smi, python, curl, trtexec) are now
 called in a stderr-safe way. TensorRT prints its progress to STDERR
 which Windows PowerShell previously treated as a fatal error - that
 was the cause of "NativeCommandError" during engine building.
=====================================================================
#>
param(
    [string]$Base = 'C:\AIVideoEnhancer',
    [switch]$Minimal   # forward to build-engines.ps1: skip the heavy x2plus engine
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }

$ModelsDir   = Join-Path $Base 'models'
$EnginesDir  = Join-Path $Base 'engines'
$LogsDir     = Join-Path $Base 'logs'
$MlrtDir     = Join-Path $Base 'vs-mlrt'
$MpvConfDir  = Join-Path $Base 'mpv-config'
$ScriptsDir  = Join-Path $Base 'scripts'

function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "    [OK] $m" -ForegroundColor Green }
function Write-Wrn($m)  { Write-Host "    [!]  $m" -ForegroundColor Yellow }
function Write-Bad($m)  { Write-Host "    [X]  $m" -ForegroundColor Red }

Write-Host ''
Write-Host '=====================================================' -ForegroundColor Cyan
Write-Host ' AI Video Enhancer installer (Real-ESRGAN realtime)'    -ForegroundColor Cyan
Write-Host '=====================================================' -ForegroundColor Cyan
Write-Host "    project : $ProjectRoot"
Write-Host "    runtime : $Base"
Write-Host ''

# ---------------------------------------------------------------------
# 0. Admin + GPU (both tolerant)
# ---------------------------------------------------------------------
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Wrn 'Not running as Administrator - installing VapourSynth or writing to C:\ may fail.'
    Write-Wrn 'If something fails below, re-run from an elevated console.'
}

Write-Step 'Checking NVIDIA GPU and driver'
$gpuName = $null
try {
    $out = cmd /c 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul'
    if ($LASTEXITCODE -eq 0 -and $out) {
        $parts = ($out | Select-Object -First 1).Split(',')
        $gpuName = $parts[0].Trim()
        Write-Ok ("GPU: {0}  (driver {1})" -f $parts[0].Trim(), $parts[1].Trim())
    }
} catch { }
if (-not $gpuName) {
    Write-Wrn 'Could not query NVIDIA driver. AI inference needs a working NVIDIA driver!'
}

# ---------------------------------------------------------------------
# helper: safe downloads via curl.exe (never through PowerShell stderr)
# ---------------------------------------------------------------------
function Get-Url {
    param([string]$Url, [string]$OutFile, [int]$TimeoutSec = 7200)
    $tmp = "$OutFile.part"
    try {
        cmd /c "`"$env:SystemRoot\System32\curl.exe`" -L --fail --retry 3 --retry-delay 2 -o `"$tmp`" --connect-timeout 30 --max-time $TimeoutSec `"$Url`" 2>nul"
        if ($LASTEXITCODE -eq 0 -and (Test-Path $tmp)) {
            Move-Item $tmp $OutFile -Force
            return $true
        }
    } catch { }
    if (Test-Path $tmp) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }
    return $false
}

# ---------------------------------------------------------------------
# 1. Python 3.12+ (required by VapourSynth R66+)
# ---------------------------------------------------------------------
Write-Step 'Checking Python 3.12+ (required by VapourSynth)'
$pyVer = $null
try { $pyVer = cmd /c 'py -3.12 --version 2>nul' } catch { }
if (-not $pyVer) { try { $pyVer = cmd /c 'py -3 --version 2>nul' } catch { } }
if ($pyVer -match '3\.1[2-9]') {
    Write-Ok "Python $pyVer found (py)"
} else {
    Write-Wrn 'Python 3.12 not found - downloading the official installer...'
    $pySetup = Join-Path $env:TEMP 'python-3.12.10-amd64.exe'
    $ok = Get-Url 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' $pySetup
    if (-not $ok) { Write-Bad 'Could not download Python. Install Python 3.12 from python.org manually, then re-run.'; exit 1 }
    Write-Host '    Installing Python 3.12 (this takes a minute)...'
    Start-Process -FilePath $pySetup -ArgumentList '/quiet','InstallAllUsers=1','PrependPath=1','Include_test=0' -Wait
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
    try { $pyVer = cmd /c 'py -3.12 --version 2>nul' } catch { }
    if ($pyVer -match '3\.1[2-9]') { Write-Ok "Python $pyVer installed" }
    else { Write-Bad 'Python installation could not be verified - install it manually and re-run.'; exit 1 }
}

# ---------------------------------------------------------------------
# 2. VapourSynth
# ---------------------------------------------------------------------
function Find-Vspipe {
    $cands = @()
    try { $cands += (cmd /c 'where vspipe.exe 2>nul') } catch { }
    $cands += Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\Scripts\vspipe.exe" -ErrorAction SilentlyContinue |
              Select-Object -ExpandProperty FullName
    $cands += 'C:\Program Files\VapourSynth\bin\vspipe.exe'
    foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
    return $null
}

Write-Step 'Checking VapourSynth'
$vspipe = Find-Vspipe
if ($vspipe) {
    Write-Ok "VapourSynth: $vspipe"
} else {
    Write-Wrn 'VapourSynth not found - downloading VapourSynth R81...'
    $vsSetup = Join-Path $env:TEMP 'VapourSynth-x64-R81.exe'
    $ok = Get-Url 'https://github.com/vapoursynth/vapoursynth/releases/download/R81/VapourSynth-x64-R81.exe' $vsSetup
    if (-not $ok) { Write-Bad 'Could not download VapourSynth. Install it manually from vapoursynth.com, then re-run.'; exit 1 }
    Write-Host '    Installing VapourSynth (silently)...'
    Start-Process -FilePath $vsSetup -ArgumentList '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART' -Wait
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
    $vspipe = Find-Vspipe
    if ($vspipe) { Write-Ok "VapourSynth installed: $vspipe" }
    else { Write-Bad 'VapourSynth installed but vspipe.exe not found on PATH. Re-login or reboot, then re-run.'; exit 1 }
}

# ---------------------------------------------------------------------
# 3. vs-mlrt v15.13 TensorRT bundle (~1.6 GB, contains vs_mlrt.dll,
#    the TensorRT runtime and trtexec.exe)
# ---------------------------------------------------------------------
Write-Step 'Downloading/extracting vs-mlrt v15.13 (TensorRT bundle, ~1.8 GB)...'
function Find-MlrtDll {
    if (Test-Path (Join-Path $MlrtDir 'vs_mlrt.dll')) { return (Join-Path $MlrtDir 'vs_mlrt.dll') }
    if (Test-Path $MlrtDir) {
        $hit = Get-ChildItem $MlrtDir -Recurse -Filter 'vs_mlrt.dll' -ErrorAction SilentlyContinue |
               Select-Object -First 1 -ExpandProperty FullName
        if ($hit) { return $hit }
    }
    return $null
}
$mlrtDll = Find-MlrtDll
if ($mlrtDll) {
    Write-Ok 'vs-mlrt already present'
} else {
    $archive = Join-Path $Base 'vsmlrt-windows-x64-tensorrt.v15.13.7z'
    $url = 'https://github.com/AmusementClub/vs-mlrt/releases/download/v15.13/vsmlrt-windows-x64-tensorrt.v15.13.7z'
    if (-not (Test-Path $archive)) {
        Write-Host '    Downloading ~1.6 GB - this takes a while depending on your connection...'
        $ok = Get-Url $url $archive
        if (-not $ok) { Write-Bad 'Could not download vs-mlrt. Check your connection and re-run.'; exit 1 }
    } else {
        Write-Ok 'archive already downloaded'
    }
    # 7zr.exe: tiny standalone 7-zip console tool
    $sevenZip = $null
    try { $sevenZip = (cmd /c 'where 7z.exe 2>nul') | Select-Object -First 1 } catch { }
    if (-not ($sevenZip -and (Test-Path $sevenZip))) {
        $sevenZip = Join-Path $Base '7zr.exe'
        if (-not (Test-Path $sevenZip)) {
            $ok = Get-Url 'https://www.7-zip.org/a/7zr.exe' $sevenZip
            if (-not $ok) { Write-Bad 'Could not download 7zr.exe (needed to extract the bundle).'; exit 1 }
        }
    }
    Write-Host '    Extracting (up to a few minutes)...'
    $extractDir = Join-Path $Base '_mlrt_extract'
    if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue }
    Start-Process -FilePath $sevenZip -ArgumentList "x","`"$archive`"",'-y',"-o`"$extractDir`"" -Wait -NoNewWindow
    $hit = Get-ChildItem $extractDir -Recurse -Filter 'vs_mlrt.dll' -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty FullName
    if (-not $hit) { Write-Bad 'vs_mlrt.dll not found inside the archive - download may be corrupted. Delete the .7z and re-run.'; exit 1 }
    $srcRoot = Split-Path -Parent $hit
    if (Test-Path $MlrtDir) { Remove-Item $MlrtDir -Recurse -Force -ErrorAction SilentlyContinue }
    Move-Item $srcRoot $MlrtDir
    Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Ok "vs-mlrt installed to $MlrtDir"
}

# ---------------------------------------------------------------------
# 4. vstrt plugin sanity check (loads the plugin exactly the way the
#    .vpy scripts do - via std.LoadPlugin with altered DLL search path)
# ---------------------------------------------------------------------
Write-Step 'Verifying the vstrt (TensorRT) plugin loads'
if (-not (Test-Path $ScriptsDir)) { New-Item -ItemType Directory -Path $ScriptsDir -Force | Out-Null }
$sanity = Join-Path $ScriptsDir 'sanity-trt.vpy'
@"
import os
import vapoursynth as vs
from vapoursynth import core
base = r"$Base"
if not hasattr(core, 'trt'):
    cands = []
    mem = os.path.join(base, 'vsmlrt.path.txt')
    if os.path.isfile(mem):
        try:
            with open(mem, 'r', encoding='utf-8') as f: cands.append(f.read().strip())
        except Exception: pass
    cands.append(r"$MlrtDir\vs_mlrt.dll")
    for root, dirs, files in os.walk(base):
        if 'vs_mlrt.dll' in files:
            cands.append(os.path.join(root, 'vs_mlrt.dll'))
            break
    loaded = False
    for c in cands:
        if c and os.path.isfile(c):
            try:
                core.std.LoadPlugin(c); loaded = hasattr(core, 'trt')
            except Exception: pass
            if loaded:
                try:
                    with open(mem, 'w', encoding='utf-8') as f: f.write(c)
                except Exception: pass
                break
    if not loaded:
        raise RuntimeError('vs_mlrt.dll found but did not load')
if not hasattr(core, 'trt'):
    raise RuntimeError('vstrt plugin not available')
core.std.BlankClip(width=64, height=64, format=vs.RGBS, length=1).set_output()
"@ | Set-Content -Path $sanity -Encoding UTF8
$errFile = Join-Path $LogsDir 'sanity.err.log'
if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }
cmd /c "`"$vspipe`" --start 0 --end 0 `"$sanity`" - 2>`"$errFile`"" | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Ok 'vstrt plugin loads OK'
} else {
    Write-Bad 'vstrt plugin failed to load. Error output:'
    if (Test-Path $errFile) { Get-Content $errFile -Tail 12 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray } }
    exit 1
}

# ---------------------------------------------------------------------
# 5. ONNX models
# ---------------------------------------------------------------------
Write-Step 'Installing engine scripts and ONNX models'
if (-not (Test-Path $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null }

# REPO_RAW is filled in by the repository build (raw.githubusercontent.com)
# OwlMaster/AllFilesRope on HuggingFace hosts verified community conversions.
$RawBase = 'https://raw.githubusercontent.com/ketaevweb/AI-Video-Enhancer-RealESRGAN/main'
$Models = @(
    @{ Name='realesr-general-x4v3';     File='realesr-general-x4v3.onnx';
       Url="$RawBase/models/realesr-general-x4v3.onnx"
       Fallback='https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/realesr-general-x4v3.onnx' },
    @{ Name='realesr-general-wdn-x4v3'; File='realesr-general-wdn-x4v3.onnx';
       Url="$RawBase/models/realesr-general-wdn-x4v3.onnx"; Fallback='' },
    @{ Name='realesr-animevideov3';     File='realesr-animevideov3.onnx';
       Url="$RawBase/models/realesr-animevideov3.onnx"; Fallback='' },
    @{ Name='RealESRGAN_x4plus';        File='RealESRGAN_x4plus.onnx';
       Url='https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/RealESRGAN_x4plus.fp16.onnx'; Fallback='' },
    @{ Name='RealESRGAN_x2plus';        File='RealESRGAN_x2plus.onnx';
       Url='https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/RealESRGAN_x2plus.fp16.onnx'; Fallback='' }
)
foreach ($m in $Models) {
    $dest = Join-Path $ModelsDir $m.File
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 100KB)) {
        Write-Ok "model $($m.Name) exists"
        continue
    }
    $got = $false
    if ($m.Url -notlike '__*') { $got = Get-Url $m.Url $dest }
    if (-not $got -and $m.Fallback) { $got = Get-Url $m.Fallback $dest }
    if ($got -and (Test-Path $dest) -and ((Get-Item $dest).Length -gt 100KB)) {
        Write-Ok "model $($m.Name) downloaded"
    } else {
        if (Test-Path $dest) { Remove-Item $dest -Force -ErrorAction SilentlyContinue }
        Write-Wrn "model $($m.Name) could not be downloaded (offline? URL moved?)"
    }
}

# ---------------------------------------------------------------------
# 6. Deploy pipeline scripts + mpv configs + launchers
# ---------------------------------------------------------------------
Write-Step 'Installing pipeline scripts and mpv configs'
Copy-Item (Join-Path $ProjectRoot 'enhance-film.vpy')     $Base -Force
Copy-Item (Join-Path $ProjectRoot 'enhance-film-hd.vpy')  $Base -Force
Copy-Item (Join-Path $ProjectRoot 'enhance-anime.vpy')    $Base -Force
Copy-Item (Join-Path $ProjectRoot 'enhance-general.vpy')  $Base -Force
Copy-Item (Join-Path $ProjectRoot 'build-engines.ps1')    $Base -Force
if (Test-Path (Join-Path $ProjectRoot 'scripts\test-speed.ps1')) {
    Copy-Item (Join-Path $ProjectRoot 'scripts\test-speed.ps1') "$ScriptsDir\" -Force
}
if (-not (Test-Path $MpvConfDir)) { New-Item -ItemType Directory -Path $MpvConfDir -Force | Out-Null }
if (Test-Path (Join-Path $ProjectRoot 'mpv')) {
    Copy-Item (Join-Path $ProjectRoot 'mpv\*') $MpvConfDir -Force -ErrorAction SilentlyContinue
}
Write-Ok "scripts copied to $Base"
Write-Ok "mpv config examples copied to $MpvConfDir (they are NOT applied automatically)"

# launchers that work regardless of mpv config
$mpvExe = $null
try { $mpvExe = (cmd /c 'where mpv.exe 2>nul') | Select-Object -First 1 } catch { }
if (-not ($mpvExe -and (Test-Path $mpvExe))) {
    foreach ($p in @("$Base\mpv\mpv.exe", "$Base\mpv\mpv.com")) { if (Test-Path $p) { $mpvExe = $p; break } }
}
if ($mpvExe -and (Test-Path $mpvExe)) {
    Write-Ok "mpv found: $mpvExe"
} else {
    Write-Wrn 'mpv.exe not found. Get the shinchiro build: https://mpv.io/installation/'
    Write-Wrn 'Put mpv.exe anywhere on PATH (or C:\AIVideoEnhancer\mpv\) and re-run.'
    $mpvExe = 'mpv'
}
foreach ($profile in @('film','film-hd','anime','general')) {
    $bat = Join-Path $Base "watch-AI-$profile.bat"
    @"
@echo off
rem Play files with the AI enhancer ($profile profile)
"$mpvExe" --vf="vapoursynth=C\:/AIVideoEnhancer/enhance-$profile.vpy" %*
"@ | Set-Content -Path $bat -Encoding ASCII
}
Write-Ok 'launchers written: C:\AIVideoEnhancer\watch-AI-<profile>.bat'

# ---------------------------------------------------------------------
# 7. TensorRT engines
# ---------------------------------------------------------------------
Write-Step 'Building TensorRT engines (one-time, RRDB models take 5-15 min each)...'
if ($Minimal) {
    & (Join-Path $Base 'build-engines.ps1') -Base $Base -Minimal
} else {
    & (Join-Path $Base 'build-engines.ps1') -Base $Base
}

Write-Host ''
Write-Host '=====================================================' -ForegroundColor Cyan
Write-Host ' Setup complete.' -ForegroundColor Green
Write-Host '=====================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host '  How to watch:'
Write-Host '    - fastest way : double-click a movie onto C:\AIVideoEnhancer\watch-AI-film.bat'
Write-Host '    - or add the [aivh] profiles from mpv-config\mpv.conf into your mpv conf'
Write-Host '  Speed check  : powershell -ExecutionPolicy Bypass -File C:\AIVideoEnhancer\scripts\test-speed.ps1'
Write-Host '  Rebuild      : powershell -ExecutionPolicy Bypass -File C:\AIVideoEnhancer\build-engines.ps1'
Write-Host ''
cmd /c pause
