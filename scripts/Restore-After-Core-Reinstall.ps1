# Restore this FlashVSR fork after a Pinokio core upgrade or a stock reinstall.
# Safe to run while you are logged in. If webui.py is already running, it exits
# without copying files.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Restore-After-Core-Reinstall.ps1 -Auto
#   powershell -NoProfile -ExecutionPolicy Bypass -File Restore-After-Core-Reinstall.ps1 -Clone
#
# -Auto   repair an existing install. Does not create the app folder.
# -Clone  if the app folder is gone, clone mine into C:\pinokio\api\FlashVSR_plus_pinokio.git

param(
    [switch]$Auto,
    [switch]$Clone
)

$ErrorActionPreference = "Continue"
$Vault = Join-Path $env:LOCALAPPDATA "FAFO\FlashVSR-preserve"
$LogFile = Join-Path $Vault "restore.log"
$Repo = "C:\pinokio\api\FlashVSR_plus_pinokio.git"
$Mine = "https://github.com/Treyu2023/FlashVSR_plus_pinokio.git"

function Write-Log([string]$Message) {
    New-Item -ItemType Directory -Force -Path $Vault | Out-Null
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

function Resolve-GitExe {
    $cmd = Get-Command git -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in @(
        "C:\pinokio\bin\git\cmd\git.exe",
        "C:\Program Files\Git\cmd\git.exe"
    )) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Test-WebuiRunning([string]$Root) {
    # start.js launches "env\Scripts\python.exe -u webui.py", so the command
    # line often has no repo path. The venv executable path does.
    $hits = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $cmd = [string]$_.CommandLine
            $exe = [string]$_.ExecutablePath
            $isWebui = $cmd -like "*webui.py*"
            $inTree = ($cmd -like "*$Root*") -or ($exe -like "$Root*")
            $isWebui -and $inTree
        }
    return [bool]$hits
}

function Find-Repo {
    $pinokio = Join-Path $Repo "pinokio.js"
    if (Test-Path $pinokio) { return $Repo }
    $api = "C:\pinokio\api"
    if (-not (Test-Path $api)) { return $null }
    $hits = @(Get-ChildItem $api -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*FlashVSR*" -and (Test-Path (Join-Path $_.FullName "pinokio.js")) })
    if ($hits.Count -gt 0) { return $hits[0].FullName }
    return $null
}

function Test-StockInstall([string]$Root) {
    $install = Join-Path $Root "install.js"
    if (-not (Test-Path $install)) { return $true }
    $text = Get-Content -Raw -Path $install
    if ($text -match 'git clone https://github.com/ai-anchorite/FlashVSR_plus\.git') { return $true }
    $webui = Join-Path $Root "app\webui.py"
    if (-not (Test-Path $webui)) { return $true }
    $body = Get-Content -Raw -Path $webui
    if ($body -notmatch 'def run_group_therapy') { return $true }
    return $false
}

function Invoke-Prepare([string]$Root) {
    $guard = Join-Path $Root "scripts\env_guard.py"
    if (-not (Test-Path $guard)) {
        Write-Log "env_guard.py is missing in $Root"
        return 1
    }
    $venvPy = Join-Path $Root "app\env\Scripts\python.exe"
    if (Test-Path $venvPy) {
        & $venvPy $guard prepare
        return $LASTEXITCODE
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & py -3 $guard prepare
        return $LASTEXITCODE
    }
    & python $guard prepare
    return $LASTEXITCODE
}

$root = Find-Repo
if (-not $root) {
    if ($Clone) {
        $git = Resolve-GitExe
        if (-not $git) {
            Write-Log "git not found; cannot clone"
            exit 1
        }
        $parent = Split-Path $Repo -Parent
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Write-Log "cloning mine into $Repo"
        & $git clone $Mine $Repo
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $root = $Repo
    } else {
        Write-Log "FlashVSR folder missing. Install it in Pinokio, then this repair runs, or pass -Clone."
        exit 0
    }
}

if (Test-WebuiRunning $root) {
    Write-Log "webui.py is running under $root. Leaving files alone."
    exit 0
}

if (Test-StockInstall $root) {
    $git = Resolve-GitExe
    if (-not $git) {
        Write-Log "stock install at $root but git was not found"
        exit 1
    }
    Write-Log "stock install at $root; fetching mine"
    Push-Location $root
    try {
        & $git remote get-url mine 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            & $git remote add mine $Mine
        }
        & $git fetch mine
        if ($LASTEXITCODE -ne 0) {
            Write-Log "git fetch mine failed"
            exit $LASTEXITCODE
        }
        $dirty = & $git status --porcelain
        if (-not $dirty) {
            & $git checkout -B main mine/main
            Write-Log "checked out mine/main (exit $LASTEXITCODE)"
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        } else {
            Write-Log "working tree is dirty, so git was not reset. File reapply still runs."
        }
    } finally {
        Pop-Location
    }
}

$code = Invoke-Prepare $root
Write-Log "prepare exit $code"
exit $code
