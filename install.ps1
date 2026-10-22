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

 FIX v3: vs-mlrt v15 renamed the plugin DLL to vstrt.dll (vs_mlrt.dll
 was the old name and no longer exists in the archive - the v2
 installer searched for a file that is not there and aborted with
 "vs_mlrt.dll not found inside the archive"). The installer now looks
 for vstrt.dll (falling back to vs_mlrt.dll), reuses the extraction
 left by a previous failed run, verifies the 7-Zip exit code, and
 remembers the final DLL path. The TensorRT runtime DLLs live in the
 vsmlrt-cuda subfolder, so every .vpy loader now registers that folder
 via os.add_dll_directory() + PATH before LoadPlugin.

 FIX v4: the vstrt plugin of vs-mlrt is still built against VapourSynth
 API3 (vs-mlrt issue #171; true for v15.x, the v16.x test line and
 master as of 2026-10). VapourSynth R80 REMOVED API3 plugin support,
 so on R80/R81 the plugin load fails with "Plugin ... uses API 3,
 which is no longer supported". The installer now pins VapourSynth
 R79 - the last release with API3 support - automatically: pip
 downgrade for Python-installed VapourSynth, the R79 setup for the
 standalone one.

 FIX v5: running install.ps1 from inside $Base itself (the simple way:
 download everything into C:\AIVideoEnhancer and run it there) made
 "Copy-Item $Base\file $Base" fail with "cannot overwrite itself" and
 - with $ErrorActionPreference='Stop' - the installer died BEFORE the
 launchers and portable_config were written. Pipeline files are now
 SYNCED from GitHub main (source of truth) with local-copy fallback
 for offline machines, so re-running this installer always brings the
 current scripts - no more manual re-downloading of individual files.

 PERF v6 (measured on RTX 3080, 720p source): x4v3 @ 960x540 in gave
 only 7 fps in mpv; compute scales with model OUTPUT pixels, so the
 realtime profiles now run 640x360-in engines that output 2560x1440
 DIRECTLY (native 2K, no more internal 4K):
   general -> realesr-animevideov3 @ 640x360  (~50 fps, measured base:
   23 fps at 960x540), film -> x4v3 @ 640x360 (~15-20 fps detail mode),
   anime and film-hd unchanged. Launchers also pass
   --framedrop=decoder+vo so brief slowdowns no longer drift A/V.

 DELIVERY v7: the PERF v6 update never reached one user's machine -
 the reachability probe failed, the sync was skipped and the
 self-copy guard kept the OLD files, while the installer still
 printed "Setup complete" (engines: 0 built / 5 skipped of the OLD
 matrix). Fixes:
   - the network probe and all downloads now pass the WINDOWS SYSTEM
     PROXY to curl.exe (VPN/proxy clients set it in the registry and
     curl silently ignores it - on such machines GitHub looks dead
     to curl while the browser works fine);
   - Invoke-WebRequest (system TLS/proxy stack) is the fallback for
     both the probe and every file download;
   - a PIPELINE VERSION GATE verifies the deployed scripts actually
     contain the PERF v6 engine marker - if not, the installer
     refuses to print "Setup complete" and explains both fixes
     (restore network / extract the fresh zip over C:\AIVideoEnhancer);
   - build-engines.ps1 prints a full ENGINE INVENTORY with a marker
     for the 2K realtime engine, so "did it build the new ones?" is
     always visible;
   - VSScript.dll locator for pip-installed VapourSynth searched ONE
     DIRECTORY TOO HIGH (Split-Path -Parent of the python root).
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
# helper: safe downloads. curl.exe is fast but IGNORES the Windows
# system proxy (VPN/proxy clients set it in the registry - without it
# raw.githubusercontent.com is unreachable from some networks while
# the browser works fine). The proxy is therefore passed to curl
# explicitly, and Invoke-WebRequest (system TLS/proxy stack) is the
# fallback for both probing and downloading.
# ---------------------------------------------------------------------
$SysProxy = $null
try {
    $reg = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction Stop
    if ($reg.ProxyEnable -eq 1 -and $reg.ProxyServer) {
        $p = "$($reg.ProxyServer)"
        if ($p -match '=') {   # per-scheme form "http=host:port;https=host:port"
            $part = ($p -split ';' | Where-Object { $_ -match '^https?=' } | Select-Object -First 1)
            if ($part) { $p = ($part -split '=', 2)[1] }
        }
        if ($p) {
            if ($p -notmatch '^https?://') { $p = "http://$p" }
            $SysProxy = $p
        }
    }
} catch { }
if ($SysProxy) { Write-Host "    using Windows system proxy: $SysProxy" }

function Get-Url {
    param([string]$Url, [string]$OutFile, [int]$TimeoutSec = 7200)
    $tmp = "$OutFile.part"
    # attempt 1: curl.exe (with the system proxy, if one is configured)
    try {
        $proxyLine = if ($SysProxy) { "--proxy `"$SysProxy`"" } else { "" }
        cmd /c "`"$env:SystemRoot\System32\curl.exe`" -L --fail --retry 3 --retry-delay 2 $proxyLine -o `"$tmp`" --connect-timeout 30 --max-time $TimeoutSec `"$Url`" 2>nul"
        if ($LASTEXITCODE -eq 0 -and (Test-Path $tmp)) {
            Move-Item $tmp $OutFile -Force
            return $true
        }
    } catch { }
    # attempt 2: Invoke-WebRequest (system TLS/proxy stack, slower but
    # succeeds where curl gets throttled/blocked)
    try {
        $prev = $ProgressPreference; $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $tmp -TimeoutSec $TimeoutSec -UseBasicParsing | Out-Null
        $ProgressPreference = $prev
        if (Test-Path $tmp) {
            Move-Item $tmp $OutFile -Force
            return $true
        }
    } catch { }
    $ProgressPreference = 'SilentlyContinue'
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
    # FIX v4: fresh installs get R79 - the last release with API3 plugin
    # support. The vstrt plugin of vs-mlrt needs API3 (vs-mlrt issue #171).
    Write-Wrn 'VapourSynth not found - downloading VapourSynth R79...'
    $vsSetup = Join-Path $env:TEMP 'VapourSynth-x64-R79.exe'
    $ok = Get-Url 'https://github.com/vapoursynth/vapoursynth/releases/download/R79/VapourSynth-x64-R79.exe' $vsSetup
    if (-not $ok) { Write-Bad 'Could not download VapourSynth. Install it manually from vapoursynth.com, then re-run.'; exit 1 }
    Write-Host '    Installing VapourSynth (silently)...'
    Start-Process -FilePath $vsSetup -ArgumentList '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART' -Wait
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
    $vspipe = Find-Vspipe
    if ($vspipe) { Write-Ok "VapourSynth installed: $vspipe" }
    else { Write-Bad 'VapourSynth installed but vspipe.exe not found on PATH. Re-login or reboot, then re-run.'; exit 1 }
}

# FIX v4: pin VapourSynth R79 when a newer (R80+) build is detected.
# The vstrt plugin is API3-only and R80 removed API3 support, so any
# newer VapourSynth fails the sanity check below with
# "Plugin ... uses API 3, which is no longer supported".
function Get-VsVersion {
    try {
        # scan the WHOLE output: some builds put the version string not on the first line
        $txt = (cmd /c "`"$vspipe`" --version 2>nul") -join "`n"
        if ($txt -match 'R(\d+)') { return [int]$Matches[1] }
    } catch { }
    return $null
}

$vsVer = Get-VsVersion
if ($vsVer -and $vsVer -ge 80) {
    Write-Wrn ("VapourSynth R{0} is too new for the vstrt (TensorRT) plugin." -f $vsVer)
    Write-Wrn 'vstrt is still an API3 plugin and VapourSynth R80 removed API3 support'
    Write-Wrn '(vs-mlrt issue #171) - it fails with "uses API 3, which is no longer supported".'
    $isPipVapourSynth = ($vspipe -match '\\Python3\d+\\Scripts\\')
    if ($isPipVapourSynth) {
        Write-Host '    Pinning VapourSynth R79 via pip (engines and models stay intact)...'
        $pipDone = $false
        foreach ($pyCmd in @('py -3.12', 'py -3', 'python')) {
            cmd /c "$pyCmd -m pip install ""vapoursynth==79"" 2>nul"
            if ($LASTEXITCODE -eq 0) { $pipDone = $true; break }
        }
        if ($pipDone) {
            $vspipe = Find-Vspipe
            $vsVer = Get-VsVersion
            if ($vsVer -and $vsVer -lt 80) {
                Write-Ok ("VapourSynth R{0} pinned (API3 support restored)." -f $vsVer)
            } else {
                Write-Bad 'pip reported success but vspipe is still not R79.'
                Write-Host '    Try manually:  py -3.12 -m pip install "vapoursynth==79"  then re-run.'
                exit 1
            }
        } else {
            Write-Bad 'Could not downgrade VapourSynth via pip. Run manually:'
            Write-Host '      py -3.12 -m pip install "vapoursynth==79"' -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host '    Downloading the VapourSynth R79 standalone installer...'
        $vsSetup = Join-Path $env:TEMP 'VapourSynth-x64-R79.exe'
        $ok = Get-Url 'https://github.com/vapoursynth/vapoursynth/releases/download/R79/VapourSynth-x64-R79.exe' $vsSetup
        if (-not $ok) { Write-Bad 'Could not download the R79 installer. Install VapourSynth R79 manually, then re-run.'; exit 1 }
        Start-Process -FilePath $vsSetup -ArgumentList '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART' -Wait
        $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
        $vspipe = Find-Vspipe
        $vsVer = Get-VsVersion
        if ($vsVer -and $vsVer -lt 80) {
            Write-Ok ("VapourSynth R{0} pinned (API3 support restored)." -f $vsVer)
        } else {
            Write-Bad 'VapourSynth R79 did not replace the newer build automatically.'
            Write-Bad 'Uninstall VapourSynth via "Apps and Features", then re-run this installer.'
            exit 1
        }
    }
} elseif ($vsVer) {
    Write-Ok ("VapourSynth R{0} (API3-compatible, vstrt OK)" -f $vsVer)
} else {
    Write-Wrn 'Could not determine the VapourSynth version (continuing - the sanity check will decide).'
}

# ---------------------------------------------------------------------
# 3. vs-mlrt v15.13 TensorRT bundle (~1.6 GB, contains vstrt.dll,
#    the TensorRT runtime DLLs in vsmlrt-cuda\ and trtexec.exe)
# ---------------------------------------------------------------------
Write-Step 'Downloading/extracting vs-mlrt v15.13 (TensorRT bundle, ~1.8 GB)...'
# FIX v3: the v15+ plugin is vstrt.dll; vs_mlrt.dll is the legacy name.
function Find-MlrtDll {
    foreach ($name in @('vstrt.dll', 'vs_mlrt.dll')) {
        if (Test-Path (Join-Path $MlrtDir $name)) { return (Join-Path $MlrtDir $name) }
    }
    if (Test-Path $MlrtDir) {
        foreach ($name in @('vstrt.dll', 'vs_mlrt.dll')) {
            $hit = Get-ChildItem $MlrtDir -Recurse -Filter $name -ErrorAction SilentlyContinue |
                   Select-Object -First 1 -ExpandProperty FullName
            if ($hit) { return $hit }
        }
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
    $extractDir = Join-Path $Base '_mlrt_extract'

    # FIX v3: reuse a complete extraction left by a previous failed run
    $haveExtract = $false
    if (Test-Path $extractDir) {
        foreach ($name in @('vstrt.dll', 'vs_mlrt.dll')) {
            $probe = Get-ChildItem $extractDir -Recurse -Filter $name -ErrorAction SilentlyContinue |
                     Select-Object -First 1 -ExpandProperty FullName
            if ($probe) { $haveExtract = $true; break }
        }
    }

    if ($haveExtract) {
        Write-Ok 'reusing the extraction left by the previous run'
    } else {
        if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue }
        Write-Host '    Extracting (up to a few minutes)...'
        $p7z = Start-Process -FilePath $sevenZip -ArgumentList "x","`"$archive`"",'-y',"-o`"$extractDir`"" -Wait -NoNewWindow -PassThru
        if ($p7z.ExitCode -ne 0) {
            Write-Bad "Extraction failed (7-Zip exit code $($p7z.ExitCode))."
            Write-Bad 'The .7z is probably corrupted - delete it and re-run this installer.'
            exit 1
        }
    }

    # locate the plugin: vstrt.dll in v15+, vs_mlrt.dll in older releases
    $hit = $null
    foreach ($name in @('vstrt.dll', 'vs_mlrt.dll')) {
        $hit = Get-ChildItem $extractDir -Recurse -Filter $name -ErrorAction SilentlyContinue |
               Select-Object -First 1 -ExpandProperty FullName
        if ($hit) { break }
    }
    if (-not $hit) {
        Write-Bad 'vstrt.dll not found inside the archive - download may be corrupted. Delete the .7z and re-run.'
        exit 1
    }

    # bundle root = the directory that also holds the vsmlrt-cuda runtime
    # (TensorRT DLLs + trtexec.exe). Walk upwards, bounded by extractDir.
    $bundleRoot = Split-Path -Parent $hit
    $p = $bundleRoot
    while ($p -and $p.StartsWith($extractDir, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path (Join-Path $p 'vsmlrt-cuda')) { $bundleRoot = $p; break }
        $p = Split-Path -Parent $p
    }

    if (Test-Path $MlrtDir) { Remove-Item $MlrtDir -Recurse -Force -ErrorAction SilentlyContinue }
    Move-Item $bundleRoot $MlrtDir
    Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
    # remember the final DLL path so the .vpy scripts pick it up instantly
    $rel = $hit.Substring($bundleRoot.Length).TrimStart('\')
    Set-Content -Path (Join-Path $Base 'vsmlrt.path.txt') -Value (Join-Path $MlrtDir $rel) -Encoding ASCII
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
names = ('vstrt.dll', 'vs_mlrt.dll')
if not hasattr(core, 'trt'):
    cands = []
    mem = os.path.join(base, 'vsmlrt.path.txt')
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
    loaded = False
    last_err = None
    for c in cands:
        if c and os.path.isfile(c):
            libdir = os.path.join(os.path.dirname(c), 'vsmlrt-cuda')
            if os.path.isdir(libdir):
                try: os.add_dll_directory(libdir)
                except Exception: pass
                os.environ['PATH'] = libdir + os.pathsep + os.environ.get('PATH', '')
            try:
                core.std.LoadPlugin(c)
                loaded = hasattr(core, 'trt')
            except Exception as e:
                last_err = e
            if loaded:
                try:
                    with open(mem, 'w', encoding='utf-8') as f: f.write(c)
                except Exception: pass
                break
    if not loaded:
        msg = 'vstrt plugin (vstrt.dll / vs_mlrt.dll) not found or failed to load under ' + base
        if last_err is not None:
            msg += ': %r' % last_err
        raise RuntimeError(msg)
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
    if (Test-Path $errFile) {
        $errTxt = Get-Content $errFile -Raw
        if ($errTxt -match 'API 3') {
            Write-Wrn 'The error mentions API 3: your VapourSynth is newer than R79.'
            Write-Wrn 'R80+ removed API3 plugin support, but vstrt (vs-mlrt) still needs API3.'
            Write-Wrn 'Re-run this installer (FIX v4) - it pins VapourSynth R79 automatically.'
        }
    }
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
#    FIX v5: never Copy-Item a file onto itself (happens when
#    $ProjectRoot -eq $Base). Sync from GitHub main first; fall back
#    to a local copy only when GitHub is unreachable.
# ---------------------------------------------------------------------
Write-Step 'Installing pipeline scripts and mpv configs'
$RepoRaw = 'https://raw.githubusercontent.com/ketaevweb/AI-Video-Enhancer-RealESRGAN/main'

# quick reachability probe: offline machines skip the sync instantly
# instead of timing out on every single file. curl alone is NOT enough:
# it ignores the Windows system proxy, and some networks throttle
# raw.githubusercontent.com - Invoke-WebRequest goes through the
# system TLS/proxy stack and is tried as a fallback.
$Online = $false
try {
    $proxyLine = if ($SysProxy) { "--proxy `"$SysProxy`"" } else { "" }
    cmd /c "`"$env:SystemRoot\System32\curl.exe`" -s -o NUL --fail --connect-timeout 8 --max-time 20 $proxyLine `"$RepoRaw/README.md`" 2>nul"
    $Online = ($LASTEXITCODE -eq 0)
} catch { }
if (-not $Online) {
    try {
        $prev = $ProgressPreference; $ProgressPreference = 'SilentlyContinue'
        $probe = Invoke-WebRequest -Uri "$RepoRaw/README.md" -TimeoutSec 20 -UseBasicParsing
        $ProgressPreference = $prev
        $Online = ($probe.StatusCode -eq 200)
    } catch { }
}
if ($Online) {
    Write-Ok 'GitHub reachable - pipeline files will be synced from main'
} else {
    Write-Wrn 'GitHub is NOT reachable (offline, network throttling, or a proxy curl cannot see).'
    Write-Wrn 'Already-installed files will be KEPT AS-IS; the pipeline version gate will verify them.'
}

function Deploy-File([string]$Rel, [string]$DstDir) {
    $dst = Join-Path $DstDir (Split-Path $Rel -Leaf)
    $src = Join-Path $ProjectRoot $Rel
    $selfCopy = ([IO.Path]::GetFullPath($src) -ieq [IO.Path]::GetFullPath($dst))
    if ($Online -and (Get-Url "$RepoRaw/$Rel" $dst -TimeoutSec 120)) {
        Write-Ok "synced  $Rel  (latest from GitHub main)"
    } elseif (-not $selfCopy -and (Test-Path $src)) {
        Copy-Item $src $dst -Force
        Write-Ok "copied  $Rel  (local copy, no network)"
    } elseif (Test-Path $dst) {
        Write-Wrn "kept existing $dst (no network, no local source copy)"
    } else {
        Write-Bad "$Rel is missing and there is no network - re-run the installer online."
        exit 1
    }
}

Deploy-File 'enhance-film.vpy'     $Base
Deploy-File 'enhance-film-hd.vpy'  $Base
Deploy-File 'enhance-anime.vpy'    $Base
Deploy-File 'enhance-general.vpy'  $Base
Deploy-File 'build-engines.ps1'    $Base
if (-not (Test-Path $ScriptsDir)) { New-Item -ItemType Directory -Path $ScriptsDir -Force | Out-Null }
Deploy-File 'scripts/test-speed.ps1' $ScriptsDir
if (-not (Test-Path $MpvConfDir)) { New-Item -ItemType Directory -Path $MpvConfDir -Force | Out-Null }
Deploy-File 'mpv/mpv.conf'   $MpvConfDir
Deploy-File 'mpv/input.conf' $MpvConfDir
Write-Ok "scripts synced into $Base"
Write-Ok "mpv config examples synced into $MpvConfDir"

# ---------------------------------------------------------------------
# 6a. mpv: find the binary and deploy portable_config next to it.
#     mpv reads config ONLY from <mpv dir>\portable_config\ (or
#     %APPDATA%\mpv) - conf files sitting directly next to mpv.exe are
#     silently IGNORED, so hotkeys/auto-profile never loaded from there.
# ---------------------------------------------------------------------
$mpvExe = $null
foreach ($p in @("$Base\mpv\mpv.exe", "$Base\mpv\mpv.com")) {
    if (Test-Path $p) { $mpvExe = $p; break }
}
if (-not $mpvExe) {
    try { $w = (cmd /c 'where mpv.exe 2>nul') | Select-Object -First 1 } catch { }
    if ($w -and (Test-Path $w)) { $mpvExe = $w }
}
$mpvDir = "$Base\mpv"
if ($mpvExe -and (Test-Path $mpvExe)) {
    $mpvDir = Split-Path -Parent $mpvExe
    Write-Ok "mpv found: $mpvExe"
} else {
    $mpvExe = "$mpvDir\mpv.exe"
    Write-Wrn "mpv.exe not found under $mpvDir and not on PATH."
    Write-Wrn 'Get the shinchiro build: https://mpv.io/installation/'
    Write-Wrn "Put mpv.exe into $mpvDir and re-run this installer."
}

$portable = Join-Path $mpvDir 'portable_config'
New-Item -ItemType Directory -Path $portable -Force | Out-Null
# FIX v5: Deploy-File instead of Copy-Item from $ProjectRoot - a user who
# downloaded ONLY install.ps1 still gets a complete portable_config here.
Deploy-File 'mpv/mpv.conf'   $portable
Deploy-File 'mpv/input.conf' $portable
foreach ($stale in @('mpv.conf', 'input.conf')) {
    $sp = Join-Path $mpvDir $stale
    if (Test-Path $sp) {
        Remove-Item $sp -Force -ErrorAction SilentlyContinue
        Write-Ok "removed stale $mpvDir\$stale (mpv only reads portable_config\)"
    }
}
Write-Ok "mpv config deployed to $portable (hotkeys: Ctrl+a films / Ctrl+f detail / Ctrl+n anime / Ctrl+0 off)"

# ---------------------------------------------------------------------
# 6a-2. PIPELINE VERSION GATE (DELIVERY v7).
#       The installer used to print "Setup complete" even when the sync
#       silently kept OLD files (offline + self-copy guard). The user
#       then rightly wonders why nothing got faster. Verify the
#       functional marker of the current round inside the deployed
#       script and fail LOUDLY if the pipeline is stale.
# ---------------------------------------------------------------------
Write-Step 'Verifying pipeline version'
$genVpy = Join-Path $Base 'enhance-general.vpy'
$marker = $false
try { $marker = Select-String -Path $genVpy -Pattern 'QUALITY v9' -Quiet } catch { }
if ($marker) {
    Write-Ok 'pipeline version: QUALITY v9 (AI + texture restore on ALL sources; Ctrl+0 = AI-free natural view)'
} else {
    Write-Bad 'pipeline files are STALE - no QUALITY v9 marker in enhance-general.vpy.'
    Write-Host '    Old files either make live-action movies look like cartoons (PERF v6),'
    Write-Host '    skip the AI on HD sources (QUALITY v8), or crash with Resize error 3074.'
    Write-Host '    Two ways to fix:'
    Write-Host '      a) restore internet / VPN so GitHub is reachable, then re-run install.ps1'
    Write-Host '      b) download AI-Video-Enhancer-RealESRGAN.zip from the chat, extract it'
    Write-Host '         INTO C:\AIVideoEnhancer with file OVERWRITE, then re-run install.ps1'
    exit 1
}

# ---------------------------------------------------------------------
# 6b. locate VSScript.dll (the runtime mpv loads for VapourSynth).
#     The standalone installer / pip wheels do not always end up on the
#     system PATH - mpv would exit instantly and silently without it.
# ---------------------------------------------------------------------
$vsCoreDir = $null
$probeDirs = @()
if ($vspipe) {
    if ($vspipe -imatch '^(.*)\\bin\\vspipe\.exe$') {
        $probeDirs += (Join-Path $Matches[1] 'core64')
    }
    if ($vspipe -imatch '^(.*)\\Scripts\\vspipe\.exe$') {
        # pip layout: <python root>\Lib\site-packages\vapoursynth\VSScript.dll
        # DELIVERY v7 fix: the old code took Split-Path -Parent of the
        # python ROOT and probed one directory too high, so pip installs
        # always ended in "VSScript.dll was not auto-located".
        $probeDirs += (Join-Path $Matches[1] 'Lib\site-packages\vapoursynth')
    }
}
$probeDirs += "$env:ProgramFiles\VapourSynth\core64"
foreach ($c in $probeDirs) {
    if ($c -and (Test-Path (Join-Path $c 'VSScript.dll'))) { $vsCoreDir = $c; break }
}
if ($vsCoreDir) {
    Write-Ok "VapourSynth runtime DLLs: $vsCoreDir"
} else {
    Write-Wrn 'VSScript.dll was not auto-located - launchers will rely on the system PATH.'
}

# ---------------------------------------------------------------------
# 6c. launchers.
#     FIX: the launchers cd into $Base first and pass a RELATIVE script
#     path - no drive colon inside the filter value at all. A path like
#     C:/... inside --vf collides with the ':' sub-option separator of
#     the mpv filter parser; the filter fails to initialize and GUI-mpv
#     exits instantly with no visible error ("drag does nothing").
#     They also prepend the VapourSynth core dir to PATH so mpv always
#     finds VSScript.dll.
# ---------------------------------------------------------------------
$pathLine = ''
if ($vsCoreDir) {
    $pathLine = "if exist `"$vsCoreDir\VSScript.dll`" set `"PATH=$vsCoreDir;%PATH%`""
}
function Write-Bat([string]$Path, [string]$Text) {
    # normalize to CRLF: cmd.exe is picky with LF-only batch files
    $Text = $Text -replace "(?<!`r)`n", "`r`n"
    Set-Content -Path $Path -Value $Text -Encoding ASCII
}
foreach ($profile in @('film','film-hd','anime','general')) {
    $bat = Join-Path $Base "watch-AI-$profile.bat"
    Write-Bat $bat @"
@echo off
rem AI Video Enhancer - profile: $profile
rem Drag a movie onto this file (or run: watch-AI-$profile.bat "D:\Films\movie.mkv")
set "AIVH=$Base"
$pathLine
cd /d "%AIVH%"
if "%~1"=="" (
    echo Drag a movie file onto this .bat to watch it with AI enhancement.
    echo.
    pause
    exit /b 0
)
start "" "$mpvExe" --vf="vapoursynth=enhance-$profile.vpy:concurrent-frames=6" --framedrop=decoder+vo %*
"@
}
# extra launcher: general profile alias (kept for backward compatibility)
Write-Bat (Join-Path $Base 'watch-AI-general-denoise.bat') @"
@echo off
rem AI enhancer: general profile.
rem QUALITY v9: identical to watch-AI-general.bat (kept so old shortcuts
rem keep working): AI + texture restore on ALL sources; Ctrl+0 = AI-free.
set "AIVH=$Base"
$pathLine
cd /d "%AIVH%"
if "%~1"=="" (
    echo Drag a movie file onto this .bat to watch it with AI enhancement.
    echo.
    pause
    exit /b 0
)
start "" "$mpvExe" --vf="vapoursynth=enhance-general.vpy:concurrent-frames=6" --framedrop=decoder+vo %*
"@
# debug launcher: visible console + full log, for diagnosing silent exits
$mpvDebugExe = $mpvExe
if (Test-Path (Join-Path $mpvDir 'mpv.com')) { $mpvDebugExe = Join-Path $mpvDir 'mpv.com' }
Write-Bat (Join-Path $Base 'watch-AI-debug.bat') @"
@echo off
rem AI Video Enhancer - DEBUG launcher
rem If the normal watch-AI-*.bat launchers exit without a window, drag the
rem movie onto THIS file: it runs mpv with a visible console, saves a full
rem log to $LogsDir\mpv-debug.log and opens it in Notepad.
set "AIVH=$Base"
$pathLine
cd /d "%AIVH%"
echo === AI Video Enhancer debug launch ===
echo mpv : "$mpvDebugExe"
echo file: %*
echo.
"$mpvDebugExe" --vf="vapoursynth=enhance-general.vpy:concurrent-frames=6" --framedrop=decoder+vo --log-file="$LogsDir\mpv-debug.log" %*
echo.
echo === mpv exit code: %ERRORLEVEL% ===
echo log : $LogsDir\mpv-debug.log
if exist "$LogsDir\mpv-debug.log" start "" notepad "$LogsDir\mpv-debug.log"
pause
"@
Write-Ok 'launchers written: watch-AI-{general, anime, general-denoise, film, film-hd, debug}.bat'

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
Write-Host "    drag a movie onto  $Base\watch-AI-general.bat   (PRIMARY for films: AI + texture restore on HD and SD; Ctrl+0 = AI-free natural view)"
Write-Host "    or onto            $Base\watch-AI-film.bat      (opt-in detail mode ~10-15 fps, synthetic look)"
Write-Host "    or onto            $Base\watch-AI-anime.bat     (anime/animation ONLY - paints live action)"
Write-Host "    nothing happens?   drag the movie onto $Base\watch-AI-debug.bat"
Write-Host '                       (runs mpv with a visible console and opens the error log in Notepad)'
Write-Host '  In-player hotkeys: Ctrl+a films | Ctrl+f detail | Ctrl+n anime | Ctrl+0 AI off'
Write-Host '  Speed check  : powershell -ExecutionPolicy Bypass -File C:\AIVideoEnhancer\scripts\test-speed.ps1'
Write-Host '  Rebuild      : powershell -ExecutionPolicy Bypass -File C:\AIVideoEnhancer\build-engines.ps1'
Write-Host ''
cmd /c pause
