@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================
echo PPO 同屏多马里奥训练（2体，自动续训）
echo 死亡即结束；X轴长期不动判死；视角跟随存活马里奥
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

set "SAVE_DIR=checkpoints"
set "MODEL_NAME=ppo_mario_same_screen_2"
set "RESUME_MODEL="

if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM 1) 优先找 *_best.zip（如果后续产出 best 会优先使用）
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$f = Get-ChildItem -Path '%SAVE_DIR%' -File -Filter '%MODEL_NAME%*_best.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f) { $f.FullName }"`) do set "RESUME_MODEL=%%i"

REM 2) 否则找 *_final.zip（按最新修改时间）
if not defined RESUME_MODEL (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$f = Get-ChildItem -Path '%SAVE_DIR%' -File -Filter '%MODEL_NAME%*_final.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f) { $f.FullName }"`) do set "RESUME_MODEL=%%i"
)

set "COMMON_ARGS=--total-timesteps 20000 --n-envs 2 --vec-env dummy --render-mode none --overlay --overlay-agents 2 --overlay-width 800 --overlay-height 600 --overlay-fps 15 --stuck-frames 360 --model-name %MODEL_NAME%"

if defined RESUME_MODEL (
  echo [INFO] Resume from: !RESUME_MODEL!
  %PYEXE% -m ai.train_ppo %COMMON_ARGS% --resume "!RESUME_MODEL!"
) else (
  echo [INFO] No previous model found. Train from scratch.
  %PYEXE% -m ai.train_ppo %COMMON_ARGS%
)

if errorlevel 1 (
  echo [ERROR] Training failed.
  pause
  exit /b 1
)

echo [OK] Model: %SAVE_DIR%\%MODEL_NAME%_final.zip
pause
exit /b 0
