#requires -version 5.1
<#
 AI Video Enhancer - uninstaller
 Removes C:\AIVideoEnhancer (engines, models, scripts, launchers).
 Does NOT touch: VapourSynth, Python, your mpv configuration.
#>
param([string]$Base = 'C:\AIVideoEnhancer', [switch]$Yes)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Base)) {
    Write-Host "Nothing to uninstall - $Base does not exist." -ForegroundColor Yellow
    exit 0
}
if (-not $Yes) {
    $answer = Read-Host "Delete $Base completely? (engines will need a 10-20 min rebuild if you re-install) [y/N]"
    if ($answer -notmatch '^[Yy]') { Write-Host 'Cancelled.'; exit 0 }
}
Remove-Item $Base -Recurse -Force
Write-Host "Removed $Base" -ForegroundColor Green
Write-Host 'Note: VapourSynth, Python and your mpv settings were left untouched.'
