@echo off
cd /d %~dp0

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOG_DIR=logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG_FILE=%LOG_DIR%\train_cpu_%TS%.log

set MODEL=models\best\ppo_mario_ram.zip
set TIMESTEPS=200000
set SAVE_FREQ=1000
set LR=0.00025

set PY_EXE=%~dp0.venv\Scripts\python.exe
if not exist "%PY_EXE%" set PY_EXE=python

echo [TRAIN] CPU mode + save every %SAVE_FREQ% steps + lr=%LR%
echo [TRAIN] Python: %PY_EXE%
echo [TRAIN] Log file: %LOG_FILE%
echo [TRAIN] Params: device=cpu timesteps=%TIMESTEPS% save_freq=%SAVE_FREQ% lr=%LR% > "%LOG_FILE%"

if exist "%MODEL%" (
  echo [TRAIN] Resuming from %MODEL%
  "%PY_EXE%" "%~dp0train_ppo.py" --timesteps %TIMESTEPS% --device cpu --save-freq %SAVE_FREQ% --lr %LR% --resume "%MODEL%" 1>>"%LOG_FILE%" 2>&1
) else (
  echo [TRAIN] No resume model found, start from scratch.
  "%PY_EXE%" "%~dp0train_ppo.py" --timesteps %TIMESTEPS% --device cpu --save-freq %SAVE_FREQ% --lr %LR% 1>>"%LOG_FILE%" 2>&1
)

if errorlevel 1 (
  echo [TRAIN] Failed with exit code %errorlevel%. See log: %LOG_FILE%
  pause
  exit /b %errorlevel%
)

echo [TRAIN] Finished. Check log: %LOG_FILE%
pause
