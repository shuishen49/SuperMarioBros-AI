@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo PPO 同屏多马里奥训练（8体，共用一个监视窗口）
echo 说明：这不是宫格，是同一地图同屏显示多个马里奥
echo ============================================

set "PYEXE="
if exist "C:\Users\shuis\AppData\Local\Programs\Python\Python312\python.exe" set "PYEXE=C:\Users\shuis\AppData\Local\Programs\Python\Python312\python.exe"
if not defined PYEXE where python >nul 2>nul && set "PYEXE=python"
if not defined PYEXE where py >nul 2>nul && set "PYEXE=py -3"

if not defined PYEXE (
  echo [ERROR] Python not found.
  pause
  exit /b 1
)

%PYEXE% -m ai.train_ppo --total-timesteps 20000 --n-envs 8 --vec-env dummy --render-mode none --overlay --overlay-width 1280 --overlay-height 720 --overlay-fps 20 --model-name ppo_mario_same_screen_8
if errorlevel 1 (
  echo [ERROR] Training failed.
  pause
  exit /b 1
)

echo [OK] Model: checkpoints\ppo_mario_same_screen_8_final.zip
pause
exit /b 0
