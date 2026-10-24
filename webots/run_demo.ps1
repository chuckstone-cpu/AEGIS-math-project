# One-command APEX demo in Webots.
#   powershell -ExecutionPolicy Bypass -File webots\run_demo.ps1 [-Duration 200] [-Run demo] [-Watch]
# -Watch opens Webots in real time (no recording) so you can drive along live.
param(
    [double]$Duration = 200,
    [string]$Run = "demo",
    [switch]$Watch
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$webots = Join-Path $env:LOCALAPPDATA "Programs\Webots\msys64\mingw64\bin\webots.exe"
if (-not (Test-Path $webots)) { $webots = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" }
$ffmpeg = Join-Path (Split-Path $webots) "ffmpeg.exe"
$out = Join-Path $root "output\$Run"
New-Item -ItemType Directory -Force (Join-Path $out "dash") | Out-Null

Write-Host "[1/5] generating world (scenarios + map dressing)"
if ($Watch) {
    python (Join-Path $root "tools\build_world.py") --stage full --run $Run
} else {
    python (Join-Path $root "tools\build_world.py") --stage full --run $Run --duration $Duration --movie "$Run.mp4" --ego-record
}

Write-Host "[2/5] pre-fetching Webots assets (avoids the 'Downloading assets 0%' stall)"
python (Join-Path $root "tools\prefetch_assets.py") (Join-Path $root "worlds\apex_city.wbt")

Write-Host "[3/5] running Webots"
$world = Join-Path $root "worlds\apex_city.wbt"
if ($Watch) {
    & $webots --mode=realtime $world
    exit 0
}
& $webots --mode=fast --batch --stdout --stderr $world | Out-Null

Write-Host "[4/5] analysing logs"
python (Join-Path $root "tools\analyze_run.py") $out "APEX in Webots - $Run"

Write-Host "[5/5] composing side-by-side video (3D view | engine dashboard)"
python (Join-Path $root "tools\compose_video.py") $out $ffmpeg
Write-Host "done: $out"
