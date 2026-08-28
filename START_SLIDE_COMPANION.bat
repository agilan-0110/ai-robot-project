@echo off
title AI Professor Robot - Slide Companion Launcher
color 0A

echo ================================================================
echo           AI PROFESSOR ROBOT - SLIDE COMPANION LAUNCHER
echo ================================================================
echo.

:: Check if Python is installed and available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10+ and check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Find the script location
if exist "scripts\slide_companion.py" (
    set SCRIPT_PATH=scripts\slide_companion.py
) else if exist "slide_companion.py" (
    set SCRIPT_PATH=slide_companion.py
) else if exist "%~dp0scripts\slide_companion.py" (
    set SCRIPT_PATH=%~dp0scripts\slide_companion.py
) else if exist "%~dp0slide_companion.py" (
    set SCRIPT_PATH=%~dp0slide_companion.py
) else (
    echo [ERROR] Could not find slide_companion.py.
    echo Make sure this batch file is in the project folder.
    echo.
    pause
    exit /b 1
)

echo Starting Slide Companion Server...
echo.
python "%SCRIPT_PATH%"

if %errorlevel% neq 0 (
    echo.
    echo [SERVER STOPPED]
    pause
)
