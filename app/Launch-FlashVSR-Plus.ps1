# Launch FlashVSR+ with this machine's saved defaults (webui_config).
# Opens the Gradio UI in the browser once the server is ready.
$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $AppDir "env\Scripts\python.exe"
$WebUi = Join-Path $AppDir "webui.py"
$Port = 7860
$Url = "http://127.0.0.1:$Port"

if (-not (Test-Path $VenvPython)) {
    throw "FlashVSR Python env missing at $VenvPython. Install/start once from Pinokio first."
}
if (-not (Test-Path $WebUi)) {
    throw "webui.py missing at $WebUi"
}

# If already up, just open the browser.
try {
    $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
    Start-Process $Url
    Write-Host "FlashVSR+ already running -> $Url"
    exit 0
} catch {
    # start fresh
}

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"

Write-Host "Starting FlashVSR+ (RTX 4090 profile defaults)..."
Write-Host "App: $AppDir"
Write-Host "URL: $Url"

$proc = Start-Process -FilePath $VenvPython -ArgumentList @(
    "`"$WebUi`"",
    "--port", "$Port"
) -WorkingDirectory $AppDir -PassThru -WindowStyle Minimized

$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        throw "FlashVSR+ process exited early (code $($proc.ExitCode)). Check the app window/logs."
    }
    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        $ready = $true
        break
    } catch {
        Write-Host "  waiting for server... $($i * 2)s"
    }
}

if (-not $ready) {
    throw "FlashVSR+ did not become ready at $Url within 3 minutes."
}

Start-Process $Url
Write-Host ""
Write-Host "========================================"
Write-Host " FlashVSR+ ready (machine defaults loaded)"
Write-Host " $Url"
Write-Host " Hover option info text for RTX 4090 tips"
Write-Host "========================================"
