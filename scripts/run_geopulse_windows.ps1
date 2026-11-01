<#
==============================================================================
GEOPULSE WINDOWS RUNNER (NOT FOR LIVE USE UNTIL APPROVED BY OPERATOR)

Safety Restriction:
Live Telegram message delivery and Supabase writes are disabled by default.
This script defaults to safe mock mode (--mock-telegram --mock-model --once).
Live execution requires explicit operator authorization and the -Live flag.
==============================================================================
#>

[CmdletBinding()]
param (
    [switch]$Live,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\.."
Set-Location $ProjectRoot

Write-Host "[GEOPULSE] Working directory: $ProjectRoot" -ForegroundColor Cyan

# 1. Verify Virtual Environment
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonPath)) {
    Write-Error "[ERROR] Virtual environment not found at $PythonPath. Please set up the venv before running."
    exit 1
}

# 2. Validate Configuration without printing secrets
Write-Host "[GEOPULSE] Validating environment configuration..." -ForegroundColor Cyan
& $PythonPath -m geopulse.cli check-config
if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] Configuration validation failed. Ensure .env has valid syntax."
    exit 1
}

# 3. Verify Local Ollama Connectivity (Read-Only)
Write-Host "[GEOPULSE] Checking local Ollama connectivity..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
    $modelFound = $response.models | Where-Object { $_.name -like "*llama3.2:3b*" }
    if ($modelFound) {
        Write-Host "[OK] Ollama is running and model llama3.2:3b is available." -ForegroundColor Green
    } else {
        Write-Warning "[WARN] Ollama is reachable, but model llama3.2:3b was not found in tags."
    }
} catch {
    Write-Warning "[WARN] Local Ollama is unreachable at http://localhost:11434. Service will fall back or require mock model."
}

if ($CheckOnly) {
    Write-Host "[GEOPULSE] Pre-flight checks completed. Exiting (-CheckOnly specified)." -ForegroundColor Green
    exit 0
}

# 4. Execution Mode
if ($Live) {
    Write-Host ""
    Write-Host "==============================================================================" -ForegroundColor Red
    Write-Host "[ATTENTION] OPERATOR APPROVAL CHECK" -ForegroundColor Red
    Write-Host "You requested LIVE continuous execution mode." -ForegroundColor Yellow
    Write-Host "Ensure migration 006 is verified, Telegram credentials are valid," -ForegroundColor Yellow
    Write-Host "and you have explicit authorization to send live briefings." -ForegroundColor Yellow
    Write-Host "==============================================================================" -ForegroundColor Red
    Write-Host "Starting in 5 seconds (Press Ctrl+C to abort)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    Write-Host "[GEOPULSE] Starting live continuous service runner..." -ForegroundColor Green
    & $PythonPath -m geopulse.cli service run
} else {
    Write-Host ""
    Write-Host "[GEOPULSE] Running in SAFE VERIFICATION mode (--mock-telegram --mock-model --once)..." -ForegroundColor Yellow
    Write-Host "To run live continuous service after operator approval, invoke:" -ForegroundColor Gray
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_geopulse_windows.ps1 -Live" -ForegroundColor Gray
    Write-Host ""
    & $PythonPath -m geopulse.cli service run --mock-telegram --mock-model --once
}
