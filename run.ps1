$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $Root "runtime"
$UvDir = Join-Path $RuntimeDir "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$UvZip = Join-Path $RuntimeDir "uv.zip"
$UvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$PythonInstallDir = Join-Path $RuntimeDir "python"
$PythonVersion = "3.11"
$VenvPath = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$SetupMarker = Join-Path $RuntimeDir ".setup-complete"

$env:PIP_CACHE_DIR = Join-Path $Root "runtime\cache\pip"
$env:HF_HOME = Join-Path $Root "runtime\cache\huggingface"
$env:XDG_CACHE_HOME = Join-Path $Root "runtime\cache"
$env:NLTK_DATA = Join-Path $Root "runtime\nltk_data"
$env:UV_CACHE_DIR = Join-Path $Root "runtime\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = $PythonInstallDir
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_PROJECT_ENVIRONMENT = $VenvPath

New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:NLTK_DATA | Out-Null
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "input") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "model") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "data\csv_databases") | Out-Null

function Install-LocalUv {
    if (Test-Path $UvExe) {
        return
    }
    Write-Host "Installing local uv bootstrapper into $UvDir"
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    Invoke-WebRequest -Uri $UvUrl -OutFile $UvZip
    Expand-Archive -Path $UvZip -DestinationPath $UvDir -Force
    Remove-Item -Force $UvZip
}

function Install-LocalPython {
    Write-Host "Installing managed Python $PythonVersion into $PythonInstallDir"
    & $UvExe python install $PythonVersion
}

function Get-VenvPythonVersion {
    if (-not (Test-Path $VenvPython)) {
        return $null
    }
    return (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
}

Install-LocalUv
Install-LocalPython

$NeedsSetup = -not (Test-Path $SetupMarker)
$ExistingPythonVersion = Get-VenvPythonVersion
if (($null -ne $ExistingPythonVersion) -and ($ExistingPythonVersion -ne $PythonVersion)) {
    Write-Host "Existing virtual environment uses Python $ExistingPythonVersion; recreating with Python $PythonVersion."
    Remove-Item -Recurse -Force $VenvPath
    Remove-Item -Force -ErrorAction SilentlyContinue $SetupMarker
    $NeedsSetup = $true
}

if (-not (Test-Path $VenvPython)) {
    & $UvExe venv --seed --python $PythonVersion $VenvPath
    $NeedsSetup = $true
}

if ($NeedsSetup) {
    & $VenvPython (Join-Path $Root "scripts\setup_env.py") --strict-accel
    New-Item -ItemType File -Force -Path $SetupMarker | Out-Null
}

& $VenvPython -m wordpycket.main
