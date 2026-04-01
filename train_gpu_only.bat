@echo off
cd /d %~dp0
echo [TRAIN] GPU mode (cuda)
set MODEL=models\ppo_mario_retro.zip
if exist "%MODEL%" (
  echo [TRAIN] Resuming from %MODEL%
  D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\train_ppo.py --timesteps 100000 --device cuda --resume "%MODEL%"
) else (
  echo [TRAIN] No resume model found, start from scratch.
  D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\train_ppo.py --timesteps 100000 --device cuda
)
pause
