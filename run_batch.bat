@echo off
REM Launcher script for batch LoRA testing on Windows
REM Usage: run_batch.bat [--config CONFIG] [--dry-run] [--resume]

setlocal enabledelayedexpansion

REM Default config
set CONFIG=batch_config.yaml
set DRY_RUN=
set RESUME=

REM Parse arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--dry-run" (
    set DRY_RUN=--dry-run
    shift
    goto parse_args
)
if "%~1"=="--resume" (
    set RESUME=--resume
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: run_batch.bat [options]
    echo.
    echo Options:
    echo   --config FILE    Configuration file (default: batch_config.yaml)
    echo   --dry-run        Print execution plan without running
    echo   --resume         Resume from last checkpoint
    echo   --help           Show this help message
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:end_parse

echo ========================================
echo Wan2.2 Batch LoRA Testing
echo ========================================
echo Config: %CONFIG%
echo Dry run: %DRY_RUN%
echo Resume: %RESUME%
echo ========================================
echo.

REM Build command
set CMD=python batch_inference.py --config %CONFIG% %DRY_RUN% %RESUME%

REM Run with auto-restart on crash
set MAX_RESTARTS=5
set restart_count=0

:restart_loop
if %restart_count% geq %MAX_RESTARTS% (
    echo Max restart attempts reached. Exiting.
    exit /b 1
)

set /a restart_count+=1
echo Starting batch inference (attempt %restart_count%/%MAX_RESTARTS%)...
echo Command: %CMD%
echo.

REM Run and capture exit code
%CMD%
set exit_code=%ERRORLEVEL%

if %exit_code%==0 (
    echo.
    echo ========================================
    echo ✓ Batch inference completed successfully!
    echo ========================================
    exit /b 0
) else (
    echo.
    echo ========================================
    echo ✗ Batch inference failed with exit code: %exit_code%
    echo ========================================

    if defined RESUME (
        if %restart_count% lss %MAX_RESTARTS% (
            echo Auto-restarting with resume in 10 seconds...
            timeout /t 10 /nobreak
            goto restart_loop
        ) else (
            echo Max restarts reached. Exiting.
            exit /b %exit_code%
        )
    ) else (
        echo Resume not enabled. Exiting.
        exit /b %exit_code%
    )
)
