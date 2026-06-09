<#
.SYNOPSIS
    Launch the LLMind backend (FastAPI) and frontend (Next.js) together.

.DESCRIPTION
    Opens each server in its own PowerShell window so logs are visible and each
    can be stopped independently with Ctrl+C. Handles the Windows-specific setup:
      * Runs uvicorn directly (NOT `fastapi run` — its banner emoji crashes the
        cp1252 console) with UTF-8 IO.
      * Points the Next.js proxy at the backend via BACKEND_URL.
      * Runs each server from its correct subproject directory.
    If the design-space projection artifact is missing it is built first (needs
    only the local corpus index, no server).

.PARAMETER Install
    Run `uv sync` and `bun install` before launching (use on first run).

.PARAMETER NoBackend / -NoFrontend
    Launch only one side.

.PARAMETER BackendHost / -BackendPort
    Override the backend bind address (default 127.0.0.1:8000).

.EXAMPLE
    .\dev.ps1                 # start both
    .\dev.ps1 -Install        # install deps first, then start both
    .\dev.ps1 -NoFrontend     # backend only
#>

[CmdletBinding()]
param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [switch]$Install,
    [switch]$NoBackend,
    [switch]$NoFrontend
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$pythonDir = Join-Path $root "llmind-python"
$webDir = Join-Path $root "llmind-web"
$backendUrl = "http://$($BackendHost):$($BackendPort)"

function Assert-Tool($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$name' not found on PATH. $hint" -ForegroundColor Red
        exit 1
    }
}

Write-Host "LLMind dev launcher" -ForegroundColor Cyan
Write-Host "  root: $root"

if (-not $NoBackend) { Assert-Tool "uv" "Install uv: https://docs.astral.sh/uv/" }
if (-not $NoFrontend) { Assert-Tool "bun" "Install bun: https://bun.sh/" }

# ── Optional dependency install ───────────────────────────────────────────────
if ($Install) {
    if (-not $NoBackend) {
        Write-Host "`nInstalling backend deps (uv sync)..." -ForegroundColor Yellow
        Push-Location $pythonDir; uv sync; Pop-Location
    }
    if (-not $NoFrontend) {
        Write-Host "Installing frontend deps (bun install)..." -ForegroundColor Yellow
        Push-Location $webDir; bun install; Pop-Location
    }
}

# ── Ensure the design-space projection exists ─────────────────────────────────
if (-not $NoBackend) {
    $surface = Join-Path $pythonDir "data\projection\surface.json"
    $index = Join-Path $pythonDir "data\local_index.npz"
    if (-not (Test-Path $surface)) {
        if (Test-Path $index) {
            Write-Host "`nProjection artifact missing -- building it (no server needed)..." -ForegroundColor Yellow
            Push-Location $pythonDir
            uv run python database_pipeline.py project
            Pop-Location
        } else {
            Write-Host "`nNote: no projection artifact and no local index found." -ForegroundColor Yellow
            Write-Host "      The Design Space view will show 'surface unavailable' until you run:" -ForegroundColor Yellow
            Write-Host "        cd llmind-python; uv run python build_local_index.py" -ForegroundColor DarkGray
            Write-Host "        cd llmind-python; uv run python database_pipeline.py project" -ForegroundColor DarkGray
        }
    }
}

# ── Launch backend (uvicorn, UTF-8 IO, own window) ────────────────────────────
if (-not $NoBackend) {
    Write-Host "`nStarting backend -> $backendUrl" -ForegroundColor Green
    $backendCmd = "`$host.UI.RawUI.WindowTitle='LLMind backend'; " +
        "`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; " +
        "uv run uvicorn backend.main:app --host $BackendHost --port $BackendPort --reload"
    Start-Process -FilePath "powershell" `
        -ArgumentList @("-NoExit", "-Command", $backendCmd) `
        -WorkingDirectory $pythonDir
}

# ── Launch frontend (bun dev, proxy -> backend, own window) ───────────────────
if (-not $NoFrontend) {
    Write-Host "Starting frontend -> http://localhost:3000  (proxy -> $backendUrl)" -ForegroundColor Green
    $frontendCmd = "`$host.UI.RawUI.WindowTitle='LLMind frontend'; " +
        "`$env:BACKEND_URL='$backendUrl'; " +
        "bun dev"
    Start-Process -FilePath "powershell" `
        -ArgumentList @("-NoExit", "-Command", $frontendCmd) `
        -WorkingDirectory $webDir
}

Write-Host "`nLaunched. Each server runs in its own window -- Ctrl+C there to stop." -ForegroundColor Cyan
if (-not $NoFrontend) { Write-Host "  App:     http://localhost:3000/mindmap" }
if (-not $NoBackend)  { Write-Host "  API:     $backendUrl/docs" }
