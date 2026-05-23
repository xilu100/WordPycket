@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if errorlevel 1 (
    echo.
    echo WordPycket failed to start. Press any key to close.
    pause >nul
    exit /b 1
)
