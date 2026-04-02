$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host '==== Mario PPO One Click (single-thread) ====' -ForegroundColor Cyan

$pythonCmd = $null

try {
    py -3.8 --version | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = 'py -3.8'
    }
} catch {
}

if (-not $pythonCmd) {
    try {
        $pyver = python --version 2>&1
        if ($pyver -match 'Python 3\.8\.') {
            $pythonCmd = 'python'
        }
    } catch {
    }
}

if (-not $pythonCmd) {
    Write-Host 'Python 3.8 was not found.' -ForegroundColor Red
    Write-Host 'Please install Python 3.8 x64 first.' -ForegroundColor Yellow
    exit 1
}

Write-Host ('Using interpreter: ' + $pythonCmd) -ForegroundColor Green

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Creating virtual environment .venv ...' -ForegroundColor Cyan
    if ($pythonCmd -eq 'py -3.8') {
        py -3.8 -m venv .venv
    } else {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Failed to create virtual environment.' -ForegroundColor Red
        exit 1
    }
}

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host 'Virtual environment is broken: .venv\Scripts\python.exe not found.' -ForegroundColor Red
    exit 1
}

$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'

Write-Host 'Upgrading pip ...' -ForegroundColor Cyan
& $venvPython -m pip install -U pip --proxy http://127.0.0.1:7897
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Failed to upgrade pip.' -ForegroundColor Red
    exit 1
}

Write-Host 'Installing project dependencies ...' -ForegroundColor Cyan
& $venvPython -m pip install -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org --proxy http://127.0.0.1:7897 -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Failed to install requirements.' -ForegroundColor Red
    exit 1
}

Write-Host 'Checking CUDA / GPU ...' -ForegroundColor Cyan
& $venvPython .\check_device.py

$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

Write-Host ''
Write-Host 'Starting single-thread Mario PPO training ...' -ForegroundColor Green
Write-Host 'Do not enable render during training.' -ForegroundColor Yellow
& $venvPython .\train_ppo.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ''
    Write-Host 'Training finished. Run this to play:' -ForegroundColor Green
    Write-Host '.\.venv\Scripts\python.exe .\play.py --model models\best\ppo_mario_retro.zip --render' -ForegroundColor Yellow
} else {
    Write-Host 'Training failed or was interrupted.' -ForegroundColor Red
    exit $LASTEXITCODE
}
