$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    py -3.8 -m venv .venv
}
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
& $venvPython -m pip install -U pip --proxy http://127.0.0.1:7897
& $venvPython -m pip install -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org --proxy http://127.0.0.1:7897 -r requirements.txt
