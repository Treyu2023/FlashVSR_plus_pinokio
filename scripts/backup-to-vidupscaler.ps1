# Mirror live FlashVSR+ working CODE into VIDUpscaler as a recovery milestone.
# FlashVSR = newer/primary. VIDUpscaler = backup only (same product family).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\backup-to-vidupscaler.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\backup-to-vidupscaler.ps1 -Commit
#
# Skips: env, models, outputs, inputs, _temp (huge / machine-local).

[CmdletBinding()]
param(
    [string]$Source = 'C:\pinokio\api\FlashVSR_plus_pinokio.git',
    [string]$Dest = 'C:\pinokio\api\VIDUpscaler',
    [switch]$Commit,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path (Join-Path $Source 'app\webui.py'))) {
    throw "Source missing app\webui.py: $Source"
}
if (-not (Test-Path $Dest)) {
    throw "Dest missing: $Dest"
}

$stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$srcSha = 'unknown'
try {
    Push-Location $Source
    $tmp = git rev-parse --short HEAD 2>$null
    if ($tmp) { $srcSha = $tmp.Trim() }
} finally {
    Pop-Location
}

Write-Host "FlashVSR -> VIDUpscaler milestone backup"
Write-Host "  Source: $Source ($srcSha)"
Write-Host "  Dest:   $Dest"
Write-Host "  Stamp:  $stamp"

$rootFiles = @(
    'pinokio.js', 'start.js', 'update.js', 'install.js', 'reset.js',
    'link.js', 'torch.js', 'pinokio_meta.json', 'icon.png', 'README.md',
    'PRESERVE.md', 'ENVIRONMENT'
)
foreach ($f in $rootFiles) {
    $from = Join-Path $Source $f
    if (-not (Test-Path -LiteralPath $from)) { continue }
    if ($WhatIf) { Write-Host "  would copy $f"; continue }
    Copy-Item -LiteralPath $from -Destination (Join-Path $Dest $f) -Force
    Write-Host "  root: $f"
}

$srcScripts = Join-Path $Source 'scripts'
$dstScripts = Join-Path $Dest 'scripts'
if (Test-Path $srcScripts) {
    if (-not $WhatIf) {
        New-Item -ItemType Directory -Path $dstScripts -Force | Out-Null
        robocopy $srcScripts $dstScripts /E /XD __pycache__ .git /XF *.pyc /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    }
    Write-Host "  scripts/ mirrored"
}

$srcApp = Join-Path $Source 'app'
$dstApp = Join-Path $Dest 'app'
if (-not $WhatIf) {
    New-Item -ItemType Directory -Path $dstApp -Force | Out-Null
    # Selective copy: do not /MIR (keep any local dest env if present)
    robocopy $srcApp $dstApp /E /XO `
        /XD env _temp models Models outputs inputs __pycache__ .git agent-tools mcps terminals cache .cache `
        /XF *.pyc *.safetensors *.ckpt *.pth *.pt `
        /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy app failed exit $LASTEXITCODE" }
}
Write-Host "  app/ source mirrored (env/models/outputs/inputs skipped)"

$markerLines = @(
    '# VIDUpscaler backup milestone',
    '',
    '| Field | Value |',
    '|-------|-------|',
    "| Role | **Backup only** - live app is FlashVSR+ |",
    "| Live source | ``$Source`` |",
    "| Source git SHA | ``$srcSha`` |",
    "| Snapshot time | ``$stamp`` |",
    "| Machine | ``$env:COMPUTERNAME`` |",
    '',
    '## Policy',
    '',
    '- **FlashVSR_plus_pinokio** = newer / primary (edit here, push mine).',
    '- **VIDUpscaler** = working milestone mirror (recover code from here if FlashVSR breaks).',
    '- Re-run from FlashVSR:',
    '  ``powershell -ExecutionPolicy Bypass -File scripts\backup-to-vidupscaler.ps1 -Commit``',
    '',
    '## What was copied',
    '',
    '- Pinokio launcher scripts (pinokio.js, start.js, update.js, ...)',
    '- app source: webui.py, queue, naming, toolbox, src, configs',
    '- **Not** copied: env, models/Models, outputs, inputs, _temp',
    '',
    'Recover by copying source files back to FlashVSR app\ (app stopped). Reuse FlashVSR env.'
)
$marker = $markerLines -join "`n"

if (-not $WhatIf) {
    New-Item -ItemType Directory -Path (Join-Path $Dest 'milestones') -Force | Out-Null
    Set-Content -Path (Join-Path $Dest 'BACKUP_FROM_FLASHVSR.md') -Value $marker -Encoding UTF8
    Set-Content -Path (Join-Path $Dest "milestones\MILESTONE_$stamp.md") -Value $marker -Encoding UTF8
    Set-Content -Path (Join-Path $dstApp "MILESTONE_BACKUP_$stamp.txt") -Value "FlashVSR $srcSha @ $stamp" -Encoding UTF8
}
Write-Host "  markers written"

$gi = @(
    '# Runtime bulk - VIDUpscaler is a CODE milestone backup of FlashVSR',
    'app/env/',
    'app/models/',
    'app/Models/',
    'app/outputs/',
    'app/inputs/',
    'app/_temp/',
    'app/**/__pycache__/',
    'app/**/*.pyc',
    'app/**/.cache/',
    'app/.git/',
    'app/agent-tools/',
    'app/mcps/',
    'app/terminals/',
    '**/__pycache__/',
    '*.pyc',
    '.DS_Store',
    'Thumbs.db',
    'cache/',
    'logs/',
    'local-preserve/',
    '**/*.safetensors',
    '**/*.ckpt',
    '**/*.pth',
    '**/*.pt',
    'toolbox/model_rife/',
    'app/toolbox/model_rife/'
) -join "`n"

if (-not $WhatIf) {
    Set-Content -Path (Join-Path $Dest '.gitignore') -Value $gi -Encoding UTF8
}

if ($Commit -and -not $WhatIf) {
    # Nested app/.git makes outer git treat app as a submodule gitlink — remove it.
    $nestedGit = Join-Path $dstApp '.git'
    if (Test-Path -LiteralPath $nestedGit) {
        Remove-Item -LiteralPath $nestedGit -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  removed nested app/.git so source files can be tracked"
    }
    Push-Location $Dest
    try {
        git add -A
        # never keep bulk path gitlinks
        git rm -r --cached --ignore-unmatch app/Models app/models app/env app/outputs app/inputs app/_temp 2>$null | Out-Null
        $status = git status --porcelain
        if (-not $status) {
            Write-Host "  git: nothing new to commit"
        } else {
            $msg = "milestone(backup): FlashVSR $srcSha @ $stamp"
            git commit -m $msg
            Write-Host "  git: $msg"
            Write-Host "  note: do not push to upstream origin (ai-anchorite); local milestone only unless you add a personal remote"
        }
    } finally {
        Pop-Location
    }
}

Write-Host "Done. VIDUpscaler holds a source milestone of FlashVSR."
