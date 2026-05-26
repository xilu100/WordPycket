@echo off
setlocal

set "ROOT=%~dp0"
set "LOG_DIR=%ROOT%logs"
set "LAUNCHER_LOG=%LOG_DIR%\wordpycket-launcher.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

>>"%LAUNCHER_LOG%" echo.
>>"%LAUNCHER_LOG%" echo ==== WordPycket launcher %date% %time% ====

echo Starting WordPycket...
echo First launch may install Python packages and can take several minutes.
echo Detailed logs are saved in: "%LOG_DIR%"
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%run.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo WordPycket failed to start.
    echo Startup log saved in: "%LOG_DIR%"
    echo Press any key to close.
    pause >nul
    exit /b %EXIT_CODE%
)
