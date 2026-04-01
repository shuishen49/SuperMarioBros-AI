@echo off
chcp 65001 >nul
cd /d %~dp0

rem ===== 可改参数（按需改这里） =====
set MODEL=auto
set SHOW_EVERY=4
set RELOAD_SECONDS=2
set SCALE=1
set EPISODE_STEPS=4000
rem ==================================

echo [DEMO] model=%MODEL%, show_every=%SHOW_EVERY%, scale=%SCALE%
D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe D:\source_code\mario-yolo-ppo\live_show_best.py --model %MODEL% --show-every %SHOW_EVERY% --reload-seconds %RELOAD_SECONDS% --scale %SCALE% --episode-steps %EPISODE_STEPS% --device cpu

pause
