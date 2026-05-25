@echo off
setlocal

set "ROOT=%~dp0"
set "LOG_DIR=%ROOT%logs"
set "LAUNCHER_LOG=%LOG_DIR%\wordpycket-launcher.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

>>"%LAUNCHER_LOG%" echo.
>>"%LAUNCHER_LOG%" echo ==== WordPycket launcher %date% %time% ====

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%run.ps1" >>"%LAUNCHER_LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo WordPycket failed to start.
    echo Startup log saved to: "%LAUNCHER_LOG%"
    echo Press any key to close.
    pause >nul
    exit /b %EXIT_CODE%
)
