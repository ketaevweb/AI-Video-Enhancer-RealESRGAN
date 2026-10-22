#requires -version 5.1
<#
=====================================================================
 AI Video Enhancer - TensorRT engine builder (Real-ESRGAN realtime)
=====================================================================
 FIX v2: The previous version failed with "NativeCommandError" because
 trtexec.exe prints ALL of its progress/warnings to STDERR, and Windows
 PowerShell (with $ErrorActionPreference='Stop') treats redirected
 stderr of native executables as terminating errors.

 FIX v3: trtexec flags aligned with the official vstrt documentation
 (AmusementClub/vs-mlrt, vstrt/README.md): a static engine is built
 with --minShapes/--optShapes/--maxShapes all equal (input:1x3xHxW)
 and the timing cache flag is --timingCacheFile (the previous
 --shapes= / --timingCache= spellings do not exist in trtexec).

 FIX v4: bulletproof success detection. PowerShell 5.1's
 Start-Process -PassThru sometimes reports Process.ExitCode as $null
 even for a finished process - the v3 run therefore logged
 "(trtexec exit code )" and declared successful builds failed, then
 retried WITHOUT shapes, which made TensorRT silently override the
 input to a useless 1x3x1x1. Now:
   - trtexec is started through a small cmd wrapper that writes the
     real %ERRORLEVEL% into a status file (no .NET exit code involved);
   - BOTH attempts keep the correct static shape (a retry without
     shapes can only produce a broken engine and is never done);
   - each attempt gets its own out/err logs (nothing is overwritten);
   - the engine is written to <name>.part and renamed on success, and
     a <name>.ok marker proves it was built by this script version;
     unmarked engine files (e.g. left over from the broken v3 run)
     are rebuilt automatically;
   - on failure the actual [E] lines from the trtexec log are shown.
=====================================================================
#>
[CmdletBinding()]
param(
    [string]$Base = 'C:\AIVideoEnhancer',
    # Skip the heavy RRDB x2plus engine (720p HD path falls back to the 540p engine automatically)
    [switch]$Minimal,
    # Rebuild engines even if a marker-verified engine file already exists
    [switch]$Force,
    # TRANSCODE v2.3: build EXACTLY ONE engine for a specific movie
    # resolution, e.g. -EnsureShape "976x1280" (HxW, both multiples of 16).
    # The transcode runner calls this automatically the first time a movie
    # with a new resolution is transcoded; skipped if the engine exists.
    [string]$EnsureShape = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

$EnginesDir = Join-Path $Base 'engines'
$ModelsDir  = Join-Path $Base 'models'
$LogsDir    = Join-Path $Base 'logs'
foreach ($d in @($EnginesDir, $LogsDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "    [OK] $m" -ForegroundColor Green }
function Write-Wrn($m)  { Write-Host "    [!]  $m" -ForegroundColor Yellow }
function Write-Bad($m)  { Write-Host "    [X]  $m" -ForegroundColor Red }

Write-Host ''
Write-Host '=====================================================' -ForegroundColor Cyan
if ($EnsureShape) {
    Write-Host ' Building ONE shape-matched TensorRT engine (fp16)'   -ForegroundColor Cyan
} else {
    Write-Host ' Building TensorRT engines (fp16, static shapes)'      -ForegroundColor Cyan
}
Write-Host '=====================================================' -ForegroundColor Cyan

# ------------------------------------------------------------------
# 0. GPU info (tolerant - never fails the script)
# ------------------------------------------------------------------
$gpuName = $null
try {
    $out = cmd /c 'nvidia-smi --query-gpu=name --format=csv,noheader 2>nul'
    if ($LASTEXITCODE -eq 0 -and $out) { $gpuName = ($out | Select-Object -First 1).Trim() }
} catch { }
if ($gpuName) { Write-Ok "GPU: $gpuName" } else { Write-Wrn 'Could not query GPU via nvidia-smi (continuing anyway).' }
Write-Host ''

# ------------------------------------------------------------------
# 1. Locate trtexec.exe
#    The vs-mlrt TensorRT bundle ships trtexec together with the
#    TensorRT runtime DLLs, so search the runtime folder first.
# ------------------------------------------------------------------
function Find-Trtexec {
    $found = @()

    # (a) remembered location from a previous run
    $mem = Join-Path $Base 'trtexec.path.txt'
    if (Test-Path $mem) { $found += (Get-Content $mem -TotalCount 1).Trim() }

    # (b) anywhere under the runtime folder (fast, bounded)
    foreach ($root in @((Join-Path $Base 'vs-mlrt'), $Base)) {
        if (Test-Path $root) {
            $found += Get-ChildItem -Path $root -Recurse -Filter 'trtexec.exe' -ErrorAction SilentlyContinue |
                      Select-Object -ExpandProperty FullName
        }
    }

    # (c) on PATH
    try { $found += (cmd /c 'where trtexec.exe 2>nul') } catch { }

    # (d) standard NVIDIA toolkit locations
    foreach ($p in @('C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\bin')) {
        if (Test-Path $p) {
            $found += Get-ChildItem -Path $p -Filter 'trtexec.exe' -ErrorAction SilentlyContinue |
                      Select-Object -ExpandProperty FullName
        }
    }

    foreach ($c in $found) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

$trtexec = Find-Trtexec
if (-not $trtexec) {
    Write-Bad 'trtexec.exe was not found.'
    Write-Host '    It normally lives inside the vs-mlrt TensorRT bundle, e.g.:'
    Write-Host '      C:\AIVideoEnhancer\vs-mlrt\<...>\trtexec.exe'
    Write-Host '    If you installed TensorRT separately, put its \bin on PATH and re-run.'
    exit 1
}
Write-Ok "trtexec: $trtexec"
# remember for next time
Set-Content -Path (Join-Path $Base 'trtexec.path.txt') -Value $trtexec -Encoding ASCII
Write-Host ''

# ------------------------------------------------------------------
# 2. Housekeeping: stale partial engines and orphan trust markers
# ------------------------------------------------------------------
Get-ChildItem $EnginesDir -Filter '*.engine.part' -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem $EnginesDir -Filter '*.engine.ok' -ErrorAction SilentlyContinue | ForEach-Object {
    $eng = $_.FullName.Substring(0, $_.FullName.Length - 3)   # strip '.ok'
    if (-not (Test-Path $eng)) { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
}

# ------------------------------------------------------------------
# 3. Engine matrix
#    Order: compact models first (fast), heavy RRDB last.
#    Every engine: static shape 1x3xHxW, fp16.
#    -EnsureShape replaces the whole matrix with ONE x2plus entry.
# ------------------------------------------------------------------
$timingCache = Join-Path $EnginesDir 'timing.cache'
$batch       = Join-Path $LogsDir  'run_trtexec.cmd'

if ($EnsureShape) {
    if ($EnsureShape -notmatch '^(\d+)x(\d+)$') {
        Write-Bad "-EnsureShape must look like '976x1280' (HxW) - got '$EnsureShape'"
        exit 1
    }
    $sH = [int]$Matches[1]; $sW = [int]$Matches[2]
    if (($sH -lt 64) -or ($sW -lt 64) -or ($sH -gt 2160) -or ($sW -gt 4096) -or
        (($sH % 16) -ne 0) -or (($sW % 16) -ne 0)) {
        Write-Bad "-EnsureShape ${sH}x${sW} invalid: both dims must be multiples of 16, 64..2160 / 64..4096"
        exit 1
    }
    Write-Ok ("shape-matched engine requested: RealESRGAN_x2plus 1x3x{0}x{1}" -f $sH, $sW)
    $engines = @(
        @{ Model='RealESRGAN_x2plus'; Onnx='RealESRGAN_x2plus.onnx'; H=$sH; W=$sW; Scale=2;
           Note='one-time shape-matched engine for one movie resolution (transcode runner)' }
    )
} else {
$engines = @(
    @{ Model='realesr-animevideov3';      Onnx='realesr-animevideov3.onnx';      H=360; W=640;  Scale=4; Note='compact, fast 2K engine, anime look (optional, not used by realtime vpys)' },
    @{ Model='realesr-general-x4v3';      Onnx='realesr-general-x4v3.onnx';      H=360; W=640;  Scale=4; Note='compact general model - AI path of enhance-general.vpy + enhance-film.vpy' },
    @{ Model='realesr-general-x4v3';      Onnx='realesr-general-x4v3.onnx';      H=540; W=960;  Scale=4; Note='compact, legacy 4K internal size (kept for film-hd fallback)' },
    @{ Model='realesr-general-wdn-x4v3';  Onnx='realesr-general-wdn-x4v3.onnx';  H=540; W=960;  Scale=4; Note='compact, denoise-only variant (no longer used by realtime vpys)' },
    @{ Model='realesr-animevideov3';      Onnx='realesr-animevideov3.onnx';      H=540; W=960;  Scale=4; Note='compact, anime/video model (enhance-anime.vpy)' },
    @{ Model='RealESRGAN_x4plus';         Onnx='RealESRGAN_x4plus.onnx';         H=540; W=960;  Scale=4; Note='RRDB, best detail for live-action (max quality SD path)' },
    @{ Model='RealESRGAN_x2plus';         Onnx='RealESRGAN_x2plus.onnx';         H=720; W=1280; Scale=2; Note='RRDB, 720p->1440p direct path (enhance-film-hd.vpy + offline transcode-AI.bat)'; Optional=$true }
)
}

$built = 0; $skipped = 0; $failed = 0
$failedNames = @()

foreach ($e in $engines) {
    if ($e.Optional -and $Minimal) {
        Write-Host "[SKIP] $($e.Model) (-Minimal switch)" -ForegroundColor DarkGray
        continue
    }

    $onnxPath   = Join-Path $ModelsDir $e.Onnx
    $engineName = '{0}_1x3x{1}x{2}_fp16.engine' -f $e.Model, $e.H, $e.W
    $enginePath = Join-Path $EnginesDir $engineName
    $enginePart = "$enginePath.part"
    $engineMark = "$enginePath.ok"
    $shape      = 'input:1x3x{0}x{1}' -f $e.H, $e.W

    Write-Host '-------------------------------------------------------------------'
    Write-Host ('  Model : {0}  ({1})' -f $e.Model, $e.Note)
    Write-Host ('  Shape : {0}   ->   output {1}x{2}' -f $shape, ($e.W * $e.Scale), ($e.H * $e.Scale))

    # skip only engines that this script built and verified itself
    # (checked BEFORE the ONNX: a verified engine is self-sufficient,
    # the ONNX is only needed to BUILD it)
    if ((Test-Path $enginePath) -and (Test-Path $engineMark) -and (-not $Force)) {
        Write-Ok "engine already built ($engineName)"; $skipped++; continue
    }

    if (-not (Test-Path $onnxPath)) {
        Write-Bad "ONNX model missing: $onnxPath"
        Write-Host '    Re-run install.ps1 to download the models first.'
        $failed++; $failedNames += $e.Model
        continue
    }
    if (Test-Path $enginePath) {
        Write-Wrn 'engine file exists but has no trust marker (leftover from a broken build) - rebuilding it'
        Remove-Item $enginePath -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $engineMark -Force -ErrorAction SilentlyContinue
    Remove-Item $enginePart -Force -ErrorAction SilentlyContinue

    $isRRDB = $e.Model -like 'RealESRGAN*'
    $eta    = if ($isRRDB) { 'RRDB network: expect 5-15 minutes' } else { 'compact network: expect 1-5 minutes' }
    Write-Host "  Build : $eta - this window may look frozen, that is normal."

    # Both attempts keep the CORRECT static shape. A retry without shapes
    # makes TensorRT override the input to 1x3x1x1 (useless engine).
    $common = "--onnx=""$onnxPath"" --fp16 --minShapes=$shape --optShapes=$shape --maxShapes=$shape --memPoolSize=workspace:2048"
    $attempts = @(
        "$common --saveEngine=""$enginePart"" --timingCacheFile=""$timingCache""",
        "$common --saveEngine=""$enginePart"""
    )

    $ok = $false
    for ($i = 1; $i -le 2; $i++) {
        $outLog = Join-Path $LogsDir ("build_{0}.a{1}.out.log" -f $e.Model, $i)
        $errLog = Join-Path $LogsDir ("build_{0}.a{1}.err.log" -f $e.Model, $i)
        $status = Join-Path $LogsDir ("build_{0}.a{1}.code"   -f $e.Model, $i)
        foreach ($f in @($outLog, $errLog, $status)) { Remove-Item $f -Force -ErrorAction SilentlyContinue }

        if ($i -eq 2) {
            Write-Wrn 'retrying WITHOUT the timing cache (in case the cache file is stale/corrupt)...'
            Remove-Item $timingCache -Force -ErrorAction SilentlyContinue
        }

        # cmd wrapper: trtexec exit code -> status file. This does NOT rely
        # on .NET Process.ExitCode, which PowerShell 5.1 sometimes reports
        # as $null even for a finished process.
        $trtLine = '"{0}" {1} >"{2}" 2>"{3}"' -f $trtexec, $attempts[$i-1], $outLog, $errLog
        Set-Content -Path $batch -Value @("@echo off", $trtLine, "echo %ERRORLEVEL% >""$status""") -Encoding ASCII

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $p  = Start-Process -FilePath $env:ComSpec -ArgumentList "/c `"$batch`"" -WindowStyle Hidden -PassThru
        while (-not $p.HasExited -and -not (Test-Path $status)) {
            Start-Sleep -Seconds 15
            if (-not $p.HasExited -and -not (Test-Path $status)) {
                Write-Host ('    ... building {0:mm\:ss}' -f $sw.Elapsed)
            }
        }
        $p.WaitForExit()
        $sw.Stop()

        $code = $null
        if (Test-Path $status) {
            $code = (Get-Content $status -TotalCount 1 -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        }
        if (-not $code) {
            if ($null -ne $p.ExitCode) { $code = "$($p.ExitCode)" } else { $code = 'unknown' }
        }

        if ($code -eq '0' -and (Test-Path $enginePart) -and ((Get-Item $enginePart).Length -gt 1MB)) {
            Move-Item $enginePart $enginePath -Force
            Set-Content -Path $engineMark -Value ("ok {0}" -f (Get-Date -Format s)) -Encoding ASCII
            $sz = [math]::Round((Get-Item $enginePath).Length / 1MB, 1)
            Write-Ok ("engine built in {0:mm\:ss}  ({1}, {2} MB)" -f $sw.Elapsed, $engineName, $sz)
            $ok = $true
            break
        }

        Write-Wrn ("trtexec attempt {0} reported exit code {1} after {2:mm\:ss}" -f $i, $code, $sw.Elapsed)
        $errLines = @()
        foreach ($lg in @($errLog, $outLog)) {
            if (Test-Path $lg) {
                $errLines += Select-String -Path $lg -Pattern '\[E\]' -ErrorAction SilentlyContinue |
                             ForEach-Object { $_.Line }
            }
        }
        if ($errLines.Count -gt 0) {
            Write-Host '    Errors reported by trtexec:'
            $errLines | Select-Object -Last 8 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
        } else {
            Write-Host '    Last lines of the error log:'
            if (Test-Path $errLog) { Get-Content $errLog -Tail 10 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray } }
            else { Write-Host '      (no error log produced)' }
        }
    }

    if ($ok) {
        $built++
    } else {
        $failed++; $failedNames += $e.Model
        Remove-Item $enginePart -Force -ErrorAction SilentlyContinue
    }
    Write-Host ''
}

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
Write-Host '====================================================='
Write-Ok ("finished: {0} built, {1} skipped, {2} failed" -f $built, $skipped, $failed)

# ------------------------------------------------------------------
# Inventory: shows exactly which engines exist RIGHT NOW, so "did it
# build the new 2K realtime one?" is always answerable at a glance.
# ------------------------------------------------------------------
Write-Host ''
Write-Host '  Engine inventory:' -ForegroundColor Cyan
Get-ChildItem $EnginesDir -Filter '*.engine' -ErrorAction SilentlyContinue | ForEach-Object {
    $mb  = [math]::Round($_.Length / 1MB, 1)
    $tag = ''
    if ($_.Name -like 'realesr-general-x4v3_1x3x360x640*')  { $tag = '   <- 2K realtime (QUALITY v9, watch-AI-general)' }
    elseif ($_.Name -like 'RealESRGAN_x2plus*')             { $tag = '   <- offline transcode (transcode-AI.bat)' }
    elseif ($_.Name -like 'realesr-animevideov3_1x3x360x640*') { $tag = '   <- legacy PERF v6 primary (superseded by QUALITY v9)' }
    Write-Host ("    {0}  ({1} MB){2}" -f $_.Name, $mb, $tag)
}
if (-not $EnsureShape) {
    if (Test-Path (Join-Path $EnginesDir 'realesr-general-x4v3_1x3x360x640_fp16.engine')) {
        Write-Ok '2K realtime engine present - watch-AI-general.bat is ready (AI on ALL sources, QUALITY v9)'
    } else {
        Write-Wrn 'the 2K realtime engine (realesr-general-x4v3_1x3x360x640_fp16) is NOT built yet!'
        Write-Host '    If the build failed above, simply re-run this script - finished engines are skipped.'
    }
}

if ($failedNames.Count -gt 0) {
    Write-Bad ('failed engines: {0}' -f ($failedNames -join ', '))
    Write-Host '    Check the per-attempt logs in' -NoNewline; Write-Host " $LogsDir" -ForegroundColor Yellow
    Write-Host '    (build_<model>.a1.err.log / .a2.err.log - attempt 1 = with timing cache, attempt 2 = without)'
    Write-Host '    You can simply re-run this script - finished engines are skipped.'
    exit 1
}

if ($EnsureShape) {
    Write-Ok 'shape-matched engine ready - the transcode job can continue.'
    exit 0
}

Write-Host ''
Write-Host 'Next step: verify realtime speed with  scripts\test-speed.ps1' -ForegroundColor Cyan
Write-Host 'Overnight batch re-encode (saves an enhanced file): drag a movie onto transcode-AI.bat'
Write-Host ''
