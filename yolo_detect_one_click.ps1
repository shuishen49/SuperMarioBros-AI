param(
    [string]$Source = 'sample_frame.png'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host '没找到项目虚拟环境，请先运行 一键启动.bat 创建 .venv。' -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Source)) {
    Write-Host ('没找到图片: ' + $Source) -ForegroundColor Red
    Write-Host '把要检测的图片放到项目目录，或者拖图片到这个脚本上。' -ForegroundColor Yellow
    exit 1
}

& $venvPython .\yolo_detect.py --source $Source
