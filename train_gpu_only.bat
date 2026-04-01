@echo off
cd /d %~dp0
echo [TRAIN] GPU mode (cuda)
D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\train_ppo.py --timesteps 100000 --device cuda
pause
