@echo off
cd /d %~dp0

set MODEL=models\best\ppo_mario_ram.zip
set TIMESTEPS=10000000
set SAVE_FREQ=100000
set LR=0.0001

set RENDER_ARGS=
if /I "%~1"=="--render" set RENDER_ARGS=--render --render-every 4
if /I "%~1"=="--render-fast" set RENDER_ARGS=--render --render-every 1

set PY_EXE=%~dp0.venv\Scripts\python.exe
if not exist "%PY_EXE%" set PY_EXE=python

echo [TRAIN] CPU mode + save every %SAVE_FREQ% steps + lr=%LR%
echo [TRAIN] Live console mode (progress/timesteps visible)
echo [TRAIN] Model path: %MODEL%
if defined RENDER_ARGS (
  echo [TRAIN] Preview window: ON  (%RENDER_ARGS%)
) else (
  echo [TRAIN] Preview window: OFF (faster)
)
echo [TRAIN] Usage: train_cpu_only.bat [--render^|--render-fast]

if exist "%MODEL%" (
  echo [TRAIN] Resuming from %MODEL%
  "%PY_EXE%" "%~dp0train_ppo.py" --timesteps %TIMESTEPS% --device cpu --save-freq %SAVE_FREQ% --lr %LR% %RENDER_ARGS% --resume "%MODEL%"
) else (
  echo [TRAIN] No resume model found, start from scratch.
  "%PY_EXE%" "%~dp0train_ppo.py" --timesteps %TIMESTEPS% --device cpu --save-freq %SAVE_FREQ% --lr %LR% %RENDER_ARGS%
)

if errorlevel 1 (
  echo [TRAIN] Failed with exit code %errorlevel%.
  pause
  exit /b %errorlevel%
)

echo [TRAIN] Finished.
pause
