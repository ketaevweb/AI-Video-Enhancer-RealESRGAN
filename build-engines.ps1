#requires -version 5.1
<#
=====================================================================
 AI Video Enhancer - TensorRT engine builder (Real-ESRGAN realtime)
=====================================================================
 FIX v2: The previous version failed with "NativeCommandError" because
 trtexec.exe prints ALL of its progress/warnings to STDERR, and Windows
 PowerShell (with $ErrorActionPreference='Stop') treats redirected
 stderr of native executables as terminating errors.

 The message "Could not read cache from ...timing.cache" is a normal
 first-run WARNING, not a failure.

 This version runs trtexec via Start-Process with stdout/stderr
 redirected to log files and decides success ONLY by the process
 exit code. It is fully resumable: finished engines are skipped.

 FIX v3: trtexec flags aligned with the official vstrt documentation
 (AmusementClub/vs-mlrt, vstrt/README.md): a static engine is built
 with --minShapes/--optShapes/--maxShapes all equal (input:1x3xHxW)
 and the timing cache flag is --timingCacheFile (the previous
 --shapes= / --timingCache= spellings do not exist in trtexec).
=====================================================================
#>
[CmdletBinding()]
param(
    [string]$Base = 'C:\AIVideoEnhancer',
    # Skip the heavy RRDB x2plus engine (720p HD path falls back to the 540p engine automatically)
    [switch]$Minimal,
    # Rebuild engines even if a finished engine file already exists
    [switch]$Force
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
Write-Host ' Building TensorRT engines (fp16, static shapes)'      -ForegroundColor Cyan
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
# 2. Engine matrix
#    Order: compact models first (fast), heavy RRDB last.
#    Every engine: static shape 1x3xHxW, fp16.
# ------------------------------------------------------------------
$timingCache = Join-Path $EnginesDir 'timing.cache'

$engines = @(
    @{ Model='realesr-general-x4v3';      Onnx='realesr-general-x4v3.onnx';      H=540; W=960;  Note='compact, denoise-capable (film/general)' },
    @{ Model='realesr-general-wdn-x4v3';  Onnx='realesr-general-wdn-x4v3.onnx';  H=540; W=960;  Note='compact, denoise-only variant (mixed in enhance-general.vpy)' },
    @{ Model='realesr-animevideov3';      Onnx='realesr-animevideov3.onnx';      H=540; W=960;  Note='compact, anime/video model' },
    @{ Model='RealESRGAN_x4plus';         Onnx='RealESRGAN_x4plus.onnx';         H=540; W=960;  Note='RRDB, best detail for live-action (max quality SD path)' },
    @{ Model='RealESRGAN_x2plus';         Onnx='RealESRGAN_x2plus.onnx';         H=720; W=1280; Note='RRDB, 720p->1440p direct path (used by enhance-film-hd.vpy)'; Optional=$true }
)

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
    $shape      = 'input:1x3x{0}x{1}' -f $e.H, $e.W

    Write-Host '-------------------------------------------------------------------'
    Write-Host ('  Model : {0}  ({1})' -f $e.Model, $e.Note)
    Write-Host ('  Shape : {0}   ->   output {1}x{2}' -f $shape, ($e.W * 4), ($e.H * 4))

    if (-not (Test-Path $onnxPath)) {
        Write-Bad "ONNX model missing: $onnxPath"
        Write-Host '    Re-run install.ps1 to download the models first.'
        $failed++; $failedNames += $e.Model
        continue
    }

    # skip only real, previously finished engines (partial files are rebuilt)
    if ((Test-Path $enginePath) -and ((Get-Item $enginePath).Length -gt 1MB)) {
        if ($Force) { Write-Wrn 'engine exists but -Force given -> rebuilding' }
        else { Write-Ok "engine already built ($engineName)"; $skipped++; continue }
    }

    $isRRDB = $e.Model -like 'RealESRGAN*'
    $eta    = if ($isRRDB) { 'RRDB network: expect 5-15 minutes' } else { 'compact network: expect 1-5 minutes' }
    Write-Host "  Build : $eta - this window may look frozen, that is normal."
    Write-Host "  Logs  : $LogsDir\build_$($e.Model).log"

    $outLog = Join-Path $LogsDir "build_$($e.Model).out.log"
    $errLog = Join-Path $LogsDir "build_$($e.Model).err.log"
    if (Test-Path $outLog) { Remove-Item $outLog -Force -ErrorAction SilentlyContinue }
    if (Test-Path $errLog) { Remove-Item $errLog -Force -ErrorAction SilentlyContinue }

    # ---- attempt 1: with static shapes + timing cache --------------
    # (official vstrt docs: minShapes/optShapes/maxShapes + timingCacheFile;
    #  all three shapes equal -> a static engine locked to 1x3xHxW)
    $argStr = "--onnx=""$onnxPath"" --fp16 --saveEngine=""$enginePath"" " +
              "--minShapes=$shape --optShapes=$shape --maxShapes=$shape " +
              "--timingCacheFile=""$timingCache"" " +
              "--memPoolSize=workspace:2048"

    $ok = $false
    $p  = $null
    foreach ($attempt in 1, 2) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $p  = Start-Process -FilePath $trtexec -ArgumentList $argStr -NoNewWindow -PassThru `
                            -RedirectStandardOutput $outLog -RedirectStandardError $errLog
        # heartbeat while trtexec works (all its output goes to the logs)
        while (-not $p.HasExited) {
            Start-Sleep -Seconds 15
            if (-not $p.HasExited) {
                Write-Host ('    ... building {0:mm\:ss}' -f $sw.Elapsed)
            }
        }
        $p.WaitForExit()
        $sw.Stop()

        if ($p.ExitCode -eq 0) {
            $ok = $true
            Write-Ok ("engine built in {0:mm\:ss}" -f $sw.Elapsed)
            break
        }

        # attempt 2: drop --shapes (covers ONNX files that already carry static dims)
        if ($attempt -eq 1) {
            Write-Wrn 'trtexec failed, retrying once WITHOUT --shapes and WITHOUT timing cache...'
            if (Test-Path $timingCache) { Remove-Item $timingCache -Force -ErrorAction SilentlyContinue }
            $argStr = "--onnx=""$onnxPath"" --fp16 --saveEngine=""$enginePath"" --memPoolSize=workspace:2048"
            continue
        }
    }

    if ($ok) {
        $built++
        $sz = [math]::Round((Get-Item $enginePath).Length / 1MB, 1)
        Write-Ok "$engineName  ($sz MB)"
    } else {
        $failed++; $failedNames += $e.Model
        Write-Bad "Engine build FAILED for $($e.Model) (trtexec exit code $($p.ExitCode))."
        Write-Host '    Last lines of the error log:'
        if (Test-Path $errLog) {
            Get-Content $errLog -Tail 15 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
        } else {
            Write-Host '      (no error log produced)'
        }
    }
    Write-Host ''
}

# ------------------------------------------------------------------
# 3. Summary
# ------------------------------------------------------------------
Write-Host '====================================================='
Write-Ok ("finished: {0} built, {1} skipped, {2} failed" -f $built, $skipped, $failed)
if ($failedNames.Count -gt 0) {
    Write-Bad ('failed engines: {0}' -f ($failedNames -join ', '))
    Write-Host '    Check the logs in' -NoNewline; Write-Host " $LogsDir" -ForegroundColor Yellow
    Write-Host '    You can simply re-run this script - finished engines are skipped.'
    exit 1
}

Write-Host ''
Write-Host 'Next step: verify realtime speed with  scripts\test-speed.ps1' -ForegroundColor Cyan
Write-Host 'Then play a movie - AI enhancement switches on automatically for SD/HD sources.'
Write-Host ''
