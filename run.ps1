$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $Root "runtime"
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir ("wordpycket-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$UvDir = Join-Path $RuntimeDir "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$UvZip = Join-Path $RuntimeDir "uv.zip"
$UvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$PythonInstallDir = Join-Path $RuntimeDir "python"
$PythonVersion = "3.11"
$VenvPath = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$SetupMarker = Join-Path $RuntimeDir ".setup-complete"
$SetupScript = Join-Path $Root "scripts\setup_env.py"
$CheckScript = Join-Path $Root "scripts\check_env.py"
$PyprojectFile = Join-Path $Root "pyproject.toml"

$TranscriptStarted = $false
$ExitCode = 0
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
try {
    Start-Transcript -Path $LogFile -Force | Out-Null
    $TranscriptStarted = $true
    Write-Host "WordPycket startup log: $LogFile"

$env:PIP_CACHE_DIR = Join-Path $Root "runtime\cache\pip"
$env:HF_HOME = Join-Path $Root "runtime\cache\huggingface"
$env:XDG_CACHE_HOME = Join-Path $Root "runtime\cache"
$env:NLTK_DATA = Join-Path $Root "runtime\nltk_data"
$env:UV_CACHE_DIR = Join-Path $Root "runtime\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = $PythonInstallDir
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_PROJECT_ENVIRONMENT = $VenvPath
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$(Join-Path $Root "src");$env:PYTHONPATH" } else { Join-Path $Root "src" }

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
    Invoke-NativeChecked $UvExe "python" "install" $PythonVersion
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-VenvPythonVersion {
    if (-not (Test-Path $VenvPython)) {
        return $null
    }
    return (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
}

function Get-SetupFingerprint {
    $Parts = @("python=$PythonVersion", "setup=2")
    foreach ($Path in @($PyprojectFile, $SetupScript, $CheckScript)) {
        if (Test-Path $Path) {
            $Hash = (Get-FileHash -Algorithm SHA256 -Path $Path).Hash
            $Parts += "$([IO.Path]::GetFileName($Path))=$Hash"
        } else {
            $Parts += "$([IO.Path]::GetFileName($Path))=missing"
        }
    }
    return ($Parts -join "`n")
}

function Test-SetupMarker {
    param([Parameter(Mandatory = $true)][string] $Expected)
    if (-not (Test-Path $SetupMarker)) {
        return $false
    }
    return ((Get-Content -Raw -Path $SetupMarker) -eq $Expected)
}

Install-LocalUv
Install-LocalPython

$SetupFingerprint = Get-SetupFingerprint
$NeedsSetup = -not (Test-SetupMarker $SetupFingerprint)
$ExistingPythonVersion = Get-VenvPythonVersion
if (($null -ne $ExistingPythonVersion) -and ($ExistingPythonVersion -ne $PythonVersion)) {
    Write-Host "Existing virtual environment uses Python $ExistingPythonVersion; recreating with Python $PythonVersion."
    Remove-Item -Recurse -Force $VenvPath
    Remove-Item -Force -ErrorAction SilentlyContinue $SetupMarker
    $NeedsSetup = $true
}

if (-not (Test-Path $VenvPython)) {
    Invoke-NativeChecked $UvExe "venv" "--seed" "--python" $PythonVersion $VenvPath
    $NeedsSetup = $true
}

if ($NeedsSetup) {
    Invoke-NativeChecked $VenvPython $SetupScript "--strict-accel"
    Set-Content -Path $SetupMarker -Value $SetupFingerprint -NoNewline
} else {
    & $VenvPython $CheckScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Environment check found missing or outdated components; repairing environment."
        Invoke-NativeChecked $VenvPython $SetupScript "--strict-accel"
        Set-Content -Path $SetupMarker -Value $SetupFingerprint -NoNewline
    }
}

Invoke-NativeChecked $VenvPython "-m" "wordpycket.main"
} catch {
    $ExitCode = 1
    Write-Host ""
    Write-Host "WordPycket failed to start."
    Write-Host "Startup log saved to: $LogFile"
    Write-Host ""
    Write-Error ($_ | Format-List * -Force | Out-String)
} finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}

exit $ExitCode
