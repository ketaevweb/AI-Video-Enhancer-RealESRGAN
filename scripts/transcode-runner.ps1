#requires -version 5.1
<#
=====================================================================
 AI Video Enhancer - OFFLINE batch transcoding driver (AIVH TRANSCODE v2)
=====================================================================
 The realtime film-hd path (RRDB x2plus, ~5 fps on an RTX 3080) is
 impossible to watch - but it makes a perfect OVERNIGHT JOB: this
 script renders the movie through the AI frame by frame (vspipe)
 and re-encodes the result with ffmpeg into a normal MKV that plays
 at full speed with ZERO GPU load afterwards.

 Pipeline:  movie -> ffms2 -> enhance-*.vpy (TensorRT AI) -> y4m
            -> HEVC 10-bit (libx265 default, or -Encoder nvenc)
            -> <name>_AI.mkv
            (audio / subtitles / chapters are copied untouched)

 Usage (see transcode-AI.bat / transcode-AI-fast.bat):
   transcode-AI.bat "D:\Films\movie.mkv"              speed test + prompt
   transcode-AI.bat "D:\Films\movie.mkv" test         25 s sample + ETA only
   transcode-AI.bat "D:\Films\movie.mkv" full         no prompt, straight run
   transcode-AI.bat "D:\Films\movie.mkv" test fast    sample with 'fast' profile
 Profiles: hd (RRDB x2plus, max quality) | fast (x4v3) | general | anime
 Encoder :  -Encoder x265 (default, best size) | -Encoder nvenc (hardware HEVC,
            frees the CPU; the right choice for the 4K-output anime profile)
            -Cq 17 = quality (CRF for x265 / CQ for nvenc; lower = better)
 In test/ask mode the script also builds <name>_AI-COMPARE.mkv: the SAME
 25 s piece, LEFT = plain-upscaled original vs RIGHT = AI result, synced,
 so you can judge the quality before committing the night job.
 ffmpeg is downloaded automatically on first use if missing.

 NOTE ON NON-ASCII PATHS: the actual pipeline runs from a tiny .cmd
 file that references everything through %ENVIRONMENT% variables
 (AIVH_VSPIPE / AIVH_FFMPEG / AIVH_VPY / AIVH_SRC / AIVH_OUT / ...).
 Environment variables are Unicode-safe, so Russian movie filenames
 survive regardless of the console codepage - the .cmd file itself
 stays pure ASCII.
=====================================================================
#>
param(
    [string]$Base = 'C:\AIVideoEnhancer',
    [Parameter(Mandatory=$true)][string]$InputFile,
    [string]$Mode = 'ask',         # ask | test | full
    [string]$Profile = 'hd',       # hd | fast | general | anime
    [string]$Encoder = 'x265',     # x265 (default, best size) | nvenc (hardware)
    [int]$Cq = 17                  # CRF (x265) / CQ (nvenc): lower = better + bigger
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok($m)   { Write-Host "    [OK] $m" -ForegroundColor Green }
function Write-Wrn($m)  { Write-Host "    [!]  $m" -ForegroundColor Yellow }
function Write-Bad($m)  { Write-Host "    [X]  $m" -ForegroundColor Red }
function Format-Duration([double]$sec) {
    if ($sec -le 0 -or [double]::IsNaN($sec) -or [double]::IsInfinity($sec)) { return '?' }
    $ts = [TimeSpan]::FromSeconds([math]::Round($sec))
    if ($ts.TotalHours -ge 1) {
        return ('{0} h {1:D2} min' -f [int][math]::Floor($ts.TotalHours), $ts.Minutes)
    }
    if ($ts.TotalMinutes -ge 1) {
        return ('{0} min' -f [int][math]::Round($ts.TotalMinutes))
    }
    return ('{0} s' -f [int]$ts.TotalSeconds)
}

# side-by-side proof clip: LEFT = the original plainly upscaled (lanczos,
# what you see today), RIGHT = the AI-rendered result. Both from the same
# 25 s piece. Trim/fps are done in filters (no input -ss): both branches
# get an identical 0-based CFR timeline, so hstack stays frame-synced.
function Build-Compare {
    param([string]$SourceFile, [string]$SampleFile, [string]$OutFile,
          [string]$Rate, [double]$StartSec, [double]$LenSec)
    $dims = (& $ffprobe -v error -select_streams v:0 -show_entries 'stream=width,height' `
             -of csv=p=0 -- $SampleFile) -join ''
    if ($dims -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*$') { return $false }
    $w = $Matches[1]; $h = $Matches[2]
    $ci = [System.Globalization.CultureInfo]::InvariantCulture
    $ss = $StartSec.ToString($ci)
    $en = ($StartSec + $LenSec).ToString($ci)
    $fc = "[0:v]trim=start=$ss:end=$en,setpts=PTS-STARTPTS,fps=$Rate,scale=${w}:${h}:flags=lanczos,setsar=1[l];" +
          "[1:v]fps=$Rate,setsar=1[r];[l][r]hstack=shortest=1,format=yuv420p[v]"
    & $ffmpeg -hide_banner -loglevel error -nostdin -y `
        -i $SourceFile -i $SampleFile `
        -filter_complex $fc -map '[v]' -map 1:a? `
        -c:v libx264 -preset veryfast -crf 16 -c:a copy $OutFile 2>$null
    return ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $OutFile))
}

# ------------------------------------------------------------------
# pipe builder / runner.
# Primary transport is y4m (self-describing: survives any output
# geometry, incl. the film-hd 4K fallback engine and scope-ratio
# crops). If the installed ffmpeg happens to be built WITHOUT the
# y4m demuxer (rare; some distro patches do that), it automatically
# retries with a rawvideo pipe using geometry from `vspipe --info`.
# All paths travel as environment variables -> Unicode-safe.
# ------------------------------------------------------------------
function New-PipeCmdFile([string]$Path, [bool]$Fragment) {
    if ($Fragment) {
        $vsp = '"%AIVH_VSPIPE%" -c y4m -p --start %AIVH_START% --end %AIVH_END% "%AIVH_VPY%" -'
    } else {
        $vsp = '"%AIVH_VSPIPE%" -c y4m -p "%AIVH_VPY%" -'
    }
    if ($Fragment) {
        # NB: parentheses make 2> bind to the WHOLE pipeline (both stderrs
        # go to the log: vspipe progress/errors + ffmpeg warnings). Without
        # them cmd redirects only the last command of the pipe.
        # Audio is trimmed by atrim (NOT by an input -ss): with '-ss X -i src
        # -c:a copy' ffmpeg rebases the decoded video to 0 but keeps the
        # copied audio at its original X-based timestamps - the preview
        # would have audio starting X seconds late. Re-encoding 25 s of
        # audio costs nothing and guarantees a 0-based, in-sync preview.
        $line = "($vsp | `"%AIVH_FFMPEG%`" -hide_banner -loglevel warning -nostdin -y %AIVH_INARGS% -i `"%AIVH_SRC%`" %AIVH_AUDARGS% -c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv420p10le `"%AIVH_SAMPLE%`") 2>`"%AIVH_LOG%`""
    } else {
        $line = "($vsp | `"%AIVH_FFMPEG%`" -hide_banner -loglevel warning -nostdin -y -nostats %AIVH_INARGS% -i `"%AIVH_SRC%`" -map 0:v:0 -map 1:a? -map 1:s? -map_metadata 1 -map_chapters 1 -c:v %AIVH_VENC% %AIVH_VENCARGS% -c:a copy -c:s copy -max_muxing_queue_size 10240 `"%AIVH_OUT%`") 2>`"%AIVH_LOG%`""
    }
    Set-Content -Path $Path -Value @($line, 'if errorlevel 1 exit /b 1', 'exit /b 0') -Encoding ASCII
}

function Read-LogTail([string]$Path, [int]$Bytes = 16384) {
    try {
        $fs = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        try {
            if ($fs.Length -gt $Bytes) { $null = $fs.Seek(-$Bytes, 'End') }
            $len = [int]($fs.Length - $fs.Position)
            if ($len -le 0) { return '' }
            $buf = New-Object byte[] $len
            $null = $fs.Read($buf, 0, $len)
            return [Text.Encoding]::GetEncoding(1252).GetString($buf)
        } finally { $fs.Close() }
    } catch { return '' }
}

# full-run renderer with a live one-line progress readout: the cmd pipe
# runs hidden (all its stderr goes to the log), this loop polls the log
# for vspipe's 'Frame: N/M' updates and redraws %, fps, ETA, elapsed.
function Start-LivePipe([string]$CmdFile) {
    Write-Host '    live progress (Ctrl+C aborts the job):' -ForegroundColor DarkGray
    $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$CmdFile`"" -NoNewWindow -PassThru
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $lastFrame = -1; $lastTick = 0.0; $fps = 0.0; $total = $nb
    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 2500
        if ($proc.HasExited) { break }
        $tail = Read-LogTail $logFile
        $frame = -1; $vsTotal = 0
        foreach ($m in [regex]::Matches($tail, 'Frame:\s*(\d+)/(\d+)')) {
            $frame = [int]$m.Groups[1].Value; $vsTotal = [int]$m.Groups[2].Value
        }
        if ($vsTotal -gt 0) { $total = $vsTotal }
        if ($frame -lt 0) { continue }
        $now = $sw.Elapsed.TotalSeconds
        if ($lastFrame -ge 0 -and $now - $lastTick -gt 0.5) {
            $inst = ($frame - $lastFrame) / ($now - $lastTick)
            if ($fps -le 0) { $fps = $inst } else { $fps = 0.3 * $inst + 0.7 * $fps }
        }
        $lastFrame = $frame; $lastTick = $now
        $pct = 0.0; if ($total -gt 0) { $pct = 100.0 * $frame / $total }
        $filled = [int][math]::Round($pct / 5); if ($filled -lt 0) { $filled = 0 }; if ($filled -gt 20) { $filled = 20 }
        $bar = ('#' * $filled) + ('.' * (20 - $filled))
        $eta = '?'; if ($fps -gt 0.01) { $eta = Format-Duration (($total - $frame) / $fps) }
        Write-Host -NoNewline ("`r    [{0}] {1,5:F1}%  {2}/{3} frames  {4,5:F2} fps  ETA {5}  elapsed {6}   " -f `
            $bar, $pct, $frame, $total, $fps, $eta, (Format-Duration $now))
    }
    $null = $proc.WaitForExit()
    Write-Host ''
    return ($proc.ExitCode -eq 0)
}

function Invoke-Pipe([string]$CmdFile, [bool]$Fragment, [switch]$Live) {
    $env:AIVH_INARGS = '-f y4m -i -'
    $haveY4m = $false
    try { $haveY4m = (& $ffmpeg -hide_banner -demuxers 2>$null | Select-String -Quiet '\sy4m\s') } catch { }
    if (-not $haveY4m) {
        Write-Wrn 'this ffmpeg build has no y4m demuxer - using a rawvideo pipe (geometry via vspipe --info)...'
        $infoTxt = (& $vspipe --info $vpy - 2>$null) -join "`n"
        $w = 0; $h = 0; $fn = 0; $fd = 1
        if ($infoTxt -match 'Width:\s*(\d+)')  { $w  = [int]$Matches[1] }
        if ($infoTxt -match 'Height:\s*(\d+)') { $h  = [int]$Matches[1] }
        if ($infoTxt -match 'FPS:\s*(\d+)\s*/\s*(\d+)') { $fn = [int]$Matches[1]; $fd = [int]$Matches[2] }
        if ($w -le 0 -or $h -le 0 -or $fn -le 0) {
            Write-Bad 'could not determine the render output geometry (vspipe --info).'
            return $false
        }
        $env:AIVH_INARGS = "-f rawvideo -pix_fmt yuv420p10le -video_size ${w}x${h} -r $fn/$fd -i -"
    }
    New-PipeCmdFile $CmdFile $Fragment
    if ($Live) { return (Start-LivePipe $CmdFile) }
    cmd /c "`"$CmdFile`""
    return ($LASTEXITCODE -eq 0)
}

Write-Host ''
Write-Host '=====================================================' -ForegroundColor Cyan
Write-Host ' AI Video Enhancer - offline transcode (batch job)'    -ForegroundColor Cyan
Write-Host '=====================================================' -ForegroundColor Cyan

# ------------------------------------------------------------------
# 0. input file
# ------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
    Write-Bad "Input file not found: $InputFile"
    exit 1
}
$InputFile = (Get-Item -LiteralPath $InputFile).FullName
Write-Ok "source: $InputFile"

# ------------------------------------------------------------------
# 1. profile + engine pre-check (fail in seconds, not after hours)
# ------------------------------------------------------------------
$profiles = @{
    hd      = @{ Vpy='enhance-film-hd.vpy';  Suffix='_AI';       Label='RealESRGAN_x2plus RRDB, max quality (~4-6 fps on RTX 3080)' }
    fast    = @{ Vpy='enhance-film.vpy';     Suffix='_AI-fast';  Label='realesr-general-x4v3 compact (~10-20 fps, 2-3x faster than hd)' }
    general = @{ Vpy='enhance-general.vpy';  Suffix='_AI-gen';   Label='general profile: x4v3 + texture restore (~10-20 fps)' }
    anime   = @{ Vpy='enhance-anime.vpy';    Suffix='_AI-anime'; Label='animevideov3 (outputs 4K - the CPU encoder will be the bottleneck)' }
}
if (-not $profiles.ContainsKey($Profile)) { $Profile = 'hd' }
$prof = $profiles[$Profile]
$vpy = Join-Path $Base $prof.Vpy
if (-not (Test-Path -LiteralPath $vpy)) {
    Write-Bad "profile script not found: $vpy - re-run install.ps1"
    exit 1
}
Write-Ok ("profile: {0} - {1}" -f $Profile, $prof.Label)

if ($Profile -eq 'hd') {
    $x2 = Join-Path $Base 'engines\RealESRGAN_x2plus_1x3x720x1280_fp16.engine'
    $x4 = Join-Path $Base 'engines\RealESRGAN_x4plus_1x3x540x960_fp16.engine'
    if (Test-Path -LiteralPath $x2) {
        Write-Ok 'engine: RealESRGAN_x2plus 720p -> 1440p (the intended one)'
    } elseif (Test-Path -LiteralPath $x4) {
        Write-Wrn 'x2plus engine is NOT built - enhance-film-hd.vpy will fall back to the'
        Write-Wrn '540p x4plus RRDB engine: SLOWER and with a 4K output (bigger file).'
        Write-Wrn 'Better: run build-engines.ps1 (without -Minimal) first.'
    } else {
        Write-Bad 'no RRDB engine found - run build-engines.ps1 first.'
        exit 1
    }
}

# ------------------------------------------------------------------
# 2. vspipe
# ------------------------------------------------------------------
function Find-Vspipe {
    $cands = @()
    try { $cands += @(cmd /c 'where vspipe.exe 2>nul') } catch { }
    $cands += Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\Scripts\vspipe.exe" -ErrorAction SilentlyContinue |
              Select-Object -ExpandProperty FullName
    $cands += 'C:\Program Files\VapourSynth\bin\vspipe.exe'
    foreach ($c in $cands) { if ($c -and (Test-Path -LiteralPath $c)) { return $c } }
    return $null
}
$vspipe = Find-Vspipe
if (-not $vspipe) { Write-Bad 'vspipe.exe not found - re-run install.ps1'; exit 1 }
Write-Ok "VapourSynth: $vspipe"

# ------------------------------------------------------------------
# 3. ffmpeg + ffprobe (auto-download on first use)
# ------------------------------------------------------------------
$BinDir = Join-Path $Base 'bin'
function Find-Tool([string]$name) {
    $local = Join-Path $BinDir $name
    if (Test-Path -LiteralPath $local) { return $local }
    $w = @()
    try { $w = @(cmd /c "where $name 2>nul") } catch { }
    foreach ($c in $w) { if ($c -and (Test-Path -LiteralPath $c)) { return $c } }
    return $null
}
function Get-Url {   # same proxy-aware download logic as install.ps1
    param([string]$Url, [string]$OutFile)
    $SysProxy = $null
    try {
        $reg = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction Stop
        if ($reg.ProxyEnable -eq 1 -and $reg.ProxyServer) {
            $p = "$($reg.ProxyServer)"
            if ($p -match '=') {
                $part = ($p -split ';' | Where-Object { $_ -match '^https?=' } | Select-Object -First 1)
                if ($part) { $p = ($part -split '=', 2)[1] }
            }
            if ($p) { if ($p -notmatch '^https?://') { $p = "http://$p" }; $SysProxy = $p }
        }
    } catch { }
    $tmp = "$OutFile.part"
    try {
        $proxyLine = ''
        if ($SysProxy) { $proxyLine = "--proxy `"$SysProxy`"" }
        cmd /c "`"$env:SystemRoot\System32\curl.exe`" -L --fail --retry 3 --retry-delay 2 $proxyLine -o `"$tmp`" --connect-timeout 30 --max-time 3600 `"$Url`" 2>nul"
        if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $tmp)) { Move-Item -LiteralPath $tmp -Destination $OutFile -Force; return $true }
    } catch { }
    try {
        Invoke-WebRequest -Uri $Url -OutFile $tmp -TimeoutSec 3600 -UseBasicParsing | Out-Null
        if (Test-Path -LiteralPath $tmp) { Move-Item -LiteralPath $tmp -Destination $OutFile -Force; return $true }
    } catch { }
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
    return $false
}

$ffmpeg = Find-Tool 'ffmpeg.exe'
if (-not $ffmpeg) {
    Write-Wrn 'ffmpeg not found - it re-encodes the AI output into the final MKV.'
    $ans = Read-Host '    Download ffmpeg automatically now (~100 MB, one-time)? [Y/n]'
    if ($ans -match '^\s*(n|no)\s*$') {
        Write-Bad 'Cancelled. Install ffmpeg manually and put ffmpeg.exe + ffprobe.exe into:'
        Write-Host "      $BinDir"
        exit 1
    }
    if (-not (Test-Path -LiteralPath $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }
    $zip = Join-Path $env:TEMP 'ffmpeg-win64-gpl.zip'
    Write-Host '    Downloading the BtbN win64-gpl build (github releases)...'
    $ok = Get-Url 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' $zip
    if (-not $ok) {
        Write-Bad 'Download failed (offline? proxy?). Download ffmpeg from ffmpeg.org manually,'
        Write-Bad 'extract ffmpeg.exe + ffprobe.exe into:'
        Write-Host "      $BinDir"
        exit 1
    }
    Write-Host '    Extracting...'
    $tmpEx = Join-Path $env:TEMP ('ffmpeg_extract_' + [guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $zip -DestinationPath $tmpEx -Force
    foreach ($t in @('ffmpeg.exe', 'ffprobe.exe')) {
        $hit = Get-ChildItem $tmpEx -Recurse -Filter $t -ErrorAction SilentlyContinue |
               Select-Object -First 1 -ExpandProperty FullName
        if ($hit) { Copy-Item -LiteralPath $hit -Destination (Join-Path $BinDir $t) -Force }
    }
    Remove-Item $tmpEx -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
    $ffmpeg = Find-Tool 'ffmpeg.exe'
    if (-not $ffmpeg) { Write-Bad 'ffmpeg.exe was not found inside the downloaded archive?'; exit 1 }
}
$ffprobe = Find-Tool 'ffprobe.exe'
if (-not $ffprobe) { Write-Bad 'ffprobe.exe not found (must sit next to ffmpeg.exe)'; exit 1 }
Write-Ok "ffmpeg: $ffmpeg"

# ------------------------------------------------------------------
# 3b. final encoder (video codec for the FULL run; the 25 s sample is
#     always written with fast x264 so the bench measures the AI, not
#     the encoder)
# ------------------------------------------------------------------
if ($Encoder -eq 'nvenc') {
    $hasNvenc = $false
    try { $hasNvenc = (& $ffmpeg -hide_banner -encoders 2>$null | Select-String -Quiet 'hevc_nvenc') } catch { }
    if (-not $hasNvenc) {
        Write-Bad 'this ffmpeg build has no hevc_nvenc - use -Encoder x265 (default) or another ffmpeg build.'
        exit 1
    }
    Write-Ok ('encoder: hevc_nvenc CQ{0} (hardware NVENC - near-zero CPU load)' -f $Cq)
} else {
    $Encoder = 'x265'
    Write-Ok ('encoder: libx265 CRF{0} medium (CPU encode - best size/quality per bit)' -f $Cq)
    if ($Profile -eq 'anime') {
        Write-Wrn 'anime profile outputs 4K: CPU x265 will be the bottleneck (~4-6 fps).'
        Write-Wrn 'Consider -Encoder nvenc for this profile (hardware, ~3x faster total).'
    }
}

# ------------------------------------------------------------------
# 4. probe the source
# ------------------------------------------------------------------
Write-Step 'Reading source info (ffprobe)'
$raw = (& $ffprobe -v error -select_streams v:0 `
        -show_entries 'stream=r_frame_rate,duration,nb_frames,width,height:format=duration' `
        -of json -- $InputFile) -join "`n"
try { $info = $raw | ConvertFrom-Json } catch { $info = $null }
$st = $null
if ($info -and $info.streams -and @($info.streams).Count -ge 1) { $st = @($info.streams)[0] }
if (-not $st) { Write-Bad 'no video stream found in this file'; exit 1 }

$fps = 0.0
if ("$($st.r_frame_rate)" -match '^\s*(\d+)\s*/\s*(\d+)\s*$') {
    $fps = [double]$Matches[1] / [double]$Matches[2]
}
if ($fps -le 0) { Write-Bad "cannot determine the frame rate from '$($st.r_frame_rate)'"; exit 1 }

$nb = [long]0
if ("$($st.nb_frames)" -match '^(\d+)$') { $nb = [long]$Matches[1] }
$dur = 0.0
if ("$($st.duration)" -match '^(\d+(\.\d+)?)$') {
    $dur = [double]$Matches[1]
} elseif ($info.format -and "$($info.format.duration)" -match '^(\d+(\.\d+)?)$') {
    $dur = [double]$Matches[1]
}
if ($nb -le 0 -and $dur -gt 0) { $nb = [long][math]::Floor($dur * $fps + 0.5) }
if ($nb -le 0) { Write-Bad 'cannot determine the frame count (live stream?); a transcode needs a known length.'; exit 1 }
Write-Ok ("video: {0}x{1}, {2} frames, {3:F3} fps, length {4}" -f $st.width, $st.height, $nb, $fps, (Format-Duration $dur))

# ------------------------------------------------------------------
# 5. output paths + free space
# ------------------------------------------------------------------
$srcDir   = Split-Path -Parent $InputFile
$stem     = [IO.Path]::GetFileNameWithoutExtension($InputFile)
$outFile  = Join-Path $srcDir ($stem + $prof.Suffix + '.mkv')
$partFile = Join-Path $srcDir ($stem + $prof.Suffix + '.part.mkv')
$sample   = Join-Path $srcDir ($stem + $prof.Suffix + '-SAMPLE.mkv')
$LogsDir  = Join-Path $Base 'logs'
$logFile  = Join-Path $LogsDir 'transcode-last.log'
if (-not (Test-Path -LiteralPath $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }

try {
    $root   = [IO.Path]::GetPathRoot($srcDir)
    $drive  = New-Object System.IO.DriveInfo($root)
    $freeGb = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    if ($freeGb -lt 10) {
        Write-Wrn ("only {0} GB free on {1} - the finished file needs roughly 4-10 GB" -f $freeGb, $drive.Name)
    } else {
        Write-Ok ("free space on {0}: {1} GB" -f $drive.Name, $freeGb)
    }
} catch { }

# ------------------------------------------------------------------
# 6. speed test: render ~600 frames (25 s of the movie, taken from
#    the 20% mark so credits/black frames do not skew the result)
# ------------------------------------------------------------------
$probeN = [int][math]::Min(600, $nb)
$startF = [int][math]::Floor($nb * 0.20)
if ($startF + $probeN -gt $nb) { $startF = [math]::Max(0, $nb - $probeN) }
$endF   = $startF + $probeN - 1
$ssSec  = ($startF / $fps).ToString([System.Globalization.CultureInfo]::InvariantCulture)

$env:AIVH                   = $Base
$env:AIVH_SRC               = $InputFile
$env:AIVH_TRANSCODE_PROFILE = $prof.Vpy
$env:AIVH_VSPIPE            = $vspipe
$env:AIVH_FFMPEG            = $ffmpeg
$env:AIVH_VPY               = $vpy
if ($Encoder -eq 'nvenc') {
    $env:AIVH_VENC      = 'hevc_nvenc'
    $env:AIVH_VENCARGS  = "-preset p7 -tune hq -rc vbr -cq $Cq -b:v 0 -spatial-aq 1 -pix_fmt p010le"
} else {
    $env:AIVH_VENC      = 'libx265'
    $env:AIVH_VENCARGS  = "-preset medium -crf $Cq -pix_fmt yuv420p10le"
}
$env:AIVH_START             = "$startF"
$env:AIVH_END               = "$endF"
$env:AIVH_SS                = $ssSec
# sample audio: atrim window = [ss, ss + sample length + 0.25 s]; audio is
# re-encoded to AAC for the preview (see New-PipeCmdFile on why not copy).
# The whole audio segment travels as ONE variable: a source without any
# audio track must not hit the [1:a] filter at all, so for it we pass a
# plain '-map 0:v:0 -an' instead.
$ci = [System.Globalization.CultureInfo]::InvariantCulture
$aendSec = ($startF / $fps + $probeN / $fps + 0.25).ToString($ci)
$hasAudio = $false
try { $hasAudio = @(& $ffprobe -v error -select_streams a -show_entries 'stream=index' -of csv=p=0 -- $InputFile).Count -gt 0 } catch { }
if ($hasAudio) {
    $env:AIVH_AUDARGS = ('-filter_complex "[1:a]atrim=start={0}:end={1},asetpts=PTS-STARTPTS[a]" -map 0:v:0 -map "[a]" -c:a aac -b:a 192k' -f $ssSec, $aendSec)
} else {
    Write-Wrn 'source has no audio track - the sample will be silent.'
    $env:AIVH_AUDARGS = '-map 0:v:0 -an'
}
$env:AIVH_SAMPLE            = $sample
$env:AIVH_OUT               = $partFile
$env:AIVH_LOG               = $logFile
Remove-Item -LiteralPath $sample -Force -ErrorAction SilentlyContinue

Write-Step ("Speed test: AI-rendering frames {0}-{1} (~{2:F0} min into the movie, takes ~2 min)..." -f ($startF + 1), $endF, ($startF / $fps / 60))
$benchCmd = Join-Path $env:TEMP 'aivh_transcode_bench.cmd'

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$pipeOk = Invoke-Pipe $benchCmd $true
$sw.Stop()
if (-not $pipeOk -or -not (Test-Path -LiteralPath $sample)) {
    Write-Bad 'the speed test failed - see the vspipe/ffmpeg error printed above.'
    Write-Bad 'Typical causes: engine not built (run build-engines.ps1) or an unreadable source file.'
    exit 1
}
$fpsMeas = $probeN / $sw.Elapsed.TotalSeconds
$etaSec  = $nb / $fpsMeas
Write-Ok ("measured speed: {0:F2} fps  ({1:F0} s for {2} frames, engine load included)" -f $fpsMeas, $sw.Elapsed.TotalSeconds, $probeN)
Write-Host ''
Write-Host ('    >>> ESTIMATED TOTAL TIME for this movie: ~{0}  ({1} frames, {2}) <<<' -f (Format-Duration $etaSec), $nb, (Format-Duration $dur)) -ForegroundColor Yellow
Write-Host ('    sample so you can SEE the result: {0}' -f $sample)
Write-Host '    (the sample is encoded fast-and-dirty x264; the final file uses slower, better HEVC)'
Write-Host ''

# ------------------------------------------------------------------
# 6b. side-by-side comparison clip (test/ask modes only): LEFT =
#     original plainly upscaled, RIGHT = the AI result, same piece.
# ------------------------------------------------------------------
$compare = Join-Path $srcDir ($stem + $prof.Suffix + '-COMPARE.mkv')
if ($Mode -ne 'full') {
    Write-Step 'Building a side-by-side comparison (LEFT original, RIGHT AI)...'
    Remove-Item -LiteralPath $compare -Force -ErrorAction SilentlyContinue
    $rate = "$([int][math]::Round($fps))"
    try {
        $rt = (& $ffprobe -v error -select_streams v:0 -show_entries 'stream=r_frame_rate' -of csv=p=0 -- $sample) -join ''
        if ("$rt" -match '^\s*(\d+/\d+)\s*$') { $rate = $Matches[1] }
    } catch { }
    if (Build-Compare $InputFile $sample $compare $rate $ssSec ($probeN / $fps)) {
        Write-Ok "comparison: $compare"
        Write-Host '    open it and step through a few frames: left = what you watch today,' -ForegroundColor DarkGray
        Write-Host '    right = the AI-transcoded picture. Same zoom, frame-synced.'        -ForegroundColor DarkGray
    } else {
        Write-Wrn 'comparison clip failed (non-fatal) - judge by the SAMPLE file instead.'
    }
    Write-Host ''
}

# ------------------------------------------------------------------
# 7. mode handling
# ------------------------------------------------------------------
if ($Mode -eq 'test') {
    Write-Ok 'test mode: nothing else will run. Watch the SAMPLE / COMPARE files above.'
    Write-Host '    If the quality suits you, start the real job with:'                  -ForegroundColor White
    Write-Host ("      transcode-AI.bat `"{0}`" full" -f $InputFile) -ForegroundColor White
    exit 0
}
if ($Mode -eq 'ask') {
    $ans = Read-Host '    Start the FULL transcode now? [y/N]'
    if ($ans -notmatch '^\s*(y|yes)\s*$') {
        Write-Host '    Cancelled - only the small sample file was written.'
        exit 0
    }
}

if (Test-Path -LiteralPath $outFile) {
    $ans = Read-Host ("    Output already exists: {0} - overwrite it? [y/N]" -f $outFile)
    if ($ans -notmatch '^\s*(y|yes)\s*$') { Write-Host '    Cancelled.'; exit 0 }
}

# ------------------------------------------------------------------
# 8. FULL transcode (overnight job; no resume - do not close the window)
# ------------------------------------------------------------------
Write-Step 'FULL TRANSCODE started (leave the PC alone; Ctrl+C aborts, nothing is resumed)'
Write-Host '    prevent sleep for the night:  powercfg -change standby-timeout-ac 0'      -ForegroundColor DarkGray
Write-Host '    (or Power settings -> Sleep -> Never; the job itself shows live progress)'
$fullCmd = Join-Path $env:TEMP 'aivh_transcode_full.cmd'
Remove-Item -LiteralPath $partFile -Force -ErrorAction SilentlyContinue

$sw2 = [System.Diagnostics.Stopwatch]::StartNew()
$pipeOk2 = Invoke-Pipe $fullCmd $false -Live
$sw2.Stop()
if (-not $pipeOk2 -or -not (Test-Path -LiteralPath $partFile)) {
    Write-Bad ("transcode FAILED (see the errors above). Last lines of {0}:" -f $logFile)
    if (Test-Path -LiteralPath $logFile) {
        Get-Content -LiteralPath $logFile -Tail 15 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
    }
    Remove-Item -LiteralPath $partFile -Force -ErrorAction SilentlyContinue
    exit 1
}

# the pipe hides vspipe failures (ffmpeg happily encodes whatever came
# through), so verify the result actually covers the whole source
$raw2 = (& $ffprobe -v error -select_streams v:0 -show_entries 'format=duration' -of json -- $partFile) -join "`n"
$outDur = 0.0
if ($raw2 -match '"duration"\s*:\s*"(\d+(\.\d+)?)"') { $outDur = [double]$Matches[1] }
if ($dur -gt 0 -and $outDur -gt 0 -and ([math]::Abs($outDur - $dur) -gt 3)) {
    Write-Bad ("result duration {0:F1} s vs source {1:F1} s - the render aborted mid-way, file deleted." -f $outDur, $dur)
    Write-Bad 'Check the vspipe error above / in the log, fix the cause and simply start again.'
    Remove-Item -LiteralPath $partFile -Force -ErrorAction SilentlyContinue
    exit 1
}
Move-Item -LiteralPath $partFile -Destination $outFile -Force
$sizeGb = [math]::Round((Get-Item -LiteralPath $outFile).Length / 1GB, 2)
Write-Host ''
Write-Ok ("DONE: {0}" -f $outFile)
Write-Ok ("size {0} GB, wall time {1}, average {2:F2} fps" -f $sizeGb, (Format-Duration $sw2.Elapsed.TotalSeconds), ($nb / $sw2.Elapsed.TotalSeconds))
Write-Host ''
Write-Host '  Play the result with PLAIN mpv (double-click) or Ctrl+0 inside the AI launchers.' -ForegroundColor Yellow
Write-Host '  The file is ALREADY enhanced - running it through watch-AI-*.bat again would'      -ForegroundColor Yellow
Write-Host '  apply the AI a second time.'                                                       -ForegroundColor Yellow
