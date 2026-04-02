@echo off
cd /d %~dp0

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOG_DIR=logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG_FILE=%LOG_DIR%\train_%TS%.log

set MODEL=models\ppo_mario_retro.zip
set TIMESTEPS=10000000
set SAVE_FREQ=100000
set LR=0.0001

echo [TRAIN] GPU mode (cuda) + save every %SAVE_FREQ% steps + lr=%LR%
echo [TRAIN] Log file: %LOG_FILE%

if exist "%MODEL%" (
  echo [TRAIN] Resuming from %MODEL%
  D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\train_ppo.py --timesteps %TIMESTEPS% --device cuda --save-freq %SAVE_FREQ% --lr %LR% --resume "%MODEL%" 1>>"%LOG_FILE%" 2>&1
) else (
  echo [TRAIN] No resume model found, start from scratch.
  D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\train_ppo.py --timesteps %TIMESTEPS% --device cuda --save-freq %SAVE_FREQ% --lr %LR% 1>>"%LOG_FILE%" 2>&1
)

echo [TRAIN] Finished. Check log: %LOG_FILE%
pause
