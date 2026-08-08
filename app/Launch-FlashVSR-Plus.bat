@echo off
title FlashVSR+ Launcher
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Launch-FlashVSR-Plus.ps1"
if errorlevel 1 (
    echo.
    echo Launch failed. Press any key to close...
    pause >nul
    exit /b 1
)
exit /b 0
