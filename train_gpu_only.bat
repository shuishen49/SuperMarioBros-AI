@echo off
cd /d %~dp0

set MODEL=models\best\ppo_mario_ram.zip
set TIMESTEPS=10000000
set SAVE_FREQ=100000
set LR=0.0001

set RENDER_ARGS=
if /I "%~1"=="--render" set RENDER_ARGS=--render --render-every 4
if /I "%~1"=="--render-fast" set RENDER_ARGS=--render --render-every 1

echo [TRAIN] GPU mode (cuda) + save every %SAVE_FREQ% steps + lr=%LR%
echo [TRAIN] Live console mode (progress/timesteps visible)
echo [TRAIN] Model path: %MODEL%
if defined RENDER_ARGS (
  echo [TRAIN] Preview window: ON  (%RENDER_ARGS%)
) else (
  echo [TRAIN] Preview window: OFF (faster)
)
echo [TRAIN] Usage: train_gpu_only.bat [--render^|--render-fast]

if exist "%MODEL%" (
  echo [TRAIN] Resuming from %MODEL%
  D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\train_ppo.py --timesteps %TIMESTEPS% --device cuda --save-freq %SAVE_FREQ% --lr %LR% %RENDER_ARGS% --resume "%MODEL%"
) else (
  echo [TRAIN] No resume model found, start from scratch.
  D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\train_ppo.py --timesteps %TIMESTEPS% --device cuda --save-freq %SAVE_FREQ% --lr %LR% %RENDER_ARGS%
)

echo [TRAIN] Finished.
pause
