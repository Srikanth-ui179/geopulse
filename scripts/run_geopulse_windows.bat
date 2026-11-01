@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM GEOPULSE WINDOWS RUNNER (NOT FOR LIVE USE UNTIL APPROVED BY OPERATOR)
REM
REM Safety Restriction:
REM Live Telegram message delivery and Supabase writes are disabled by default.
REM This script defaults to safe mock mode (--mock-telegram --mock-model --once).
REM Live execution requires explicit operator authorization and the --live flag.
REM ==============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."
set "PROJECT_ROOT=%CD%"

echo [GEOPULSE] Working directory: %PROJECT_ROOT%

REM 1. Verify Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please create or activate the project virtualenv before running.
    exit /b 1
)

REM 2. Validate Configuration without printing secrets
echo [GEOPULSE] Validating environment configuration...
.venv\Scripts\python.exe -m geopulse.cli check-config
if errorlevel 1 (
    echo [ERROR] Configuration check failed. Ensure .env is properly configured.
    exit /b 1
)

REM 3. Verify Local Ollama Inference (Non-destructive check)
echo [GEOPULSE] Checking local Ollama connectivity...
powershell -NoProfile -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 3; $found = $r.models | Where-Object { $_.name -like '*llama3.2:3b*' }; if ($found) { Write-Host '[OK] Ollama is running and model llama3.2:3b is installed.' } else { Write-Warning '[WARN] Ollama is reachable but llama3.2:3b was not found in tags.' } } catch { Write-Warning '[WARN] Local Ollama is unreachable on localhost:11434. Local LLM summarization will fall back or require mock.' }"

REM 4. Mode Selection
if "%1"=="--check-only" (
    echo [GEOPULSE] Pre-flight checks completed. Exiting [--check-only specified].
    exit /b 0
)

if "%1"=="--live" (
    echo.
    echo ==============================================================================
    echo [ATTENTION] OPERATOR APPROVAL CHECK
    echo You requested LIVE execution mode.
    echo Ensure migration 006 is verified, Telegram credentials are valid,
    echo and you have explicit authorization to send live briefings.
    echo ==============================================================================
    echo Press Ctrl+C within 5 seconds to abort, or wait to start live service...
    timeout /t 5
    echo [GEOPULSE] Starting live continuous service runner...
    .venv\Scripts\python.exe -m geopulse.cli service run
) else (
    echo.
    echo [GEOPULSE] Running in SAFE VERIFICATION mode [--mock-telegram --mock-model --once]...
    echo To run live continuous service after operator approval, invoke:
    echo   run_geopulse_windows.bat --live
    echo.
    .venv\Scripts\python.exe -m geopulse.cli service run --mock-telegram --mock-model --once
)

endlocal
