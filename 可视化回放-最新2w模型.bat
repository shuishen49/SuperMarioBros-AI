@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo PythonSuperMario visual playback (3 episodes)
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

set "MODEL="

REM 1) 优先旧默认名
if exist "checkpoints\ppo_mario_visual_2w_final.zip" set "MODEL=checkpoints\ppo_mario_visual_2w_final.zip"

REM 2) 否则找“最新的 final 模型”
if not defined MODEL (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$f = Get-ChildItem -Path 'checkpoints' -File -Filter '*_final.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f) { $f.FullName }"`) do set "MODEL=%%i"
)

if not defined MODEL (
  echo [WARN] No model found in checkpoints\*_final.zip
  echo Run training bat first.
  pause
  exit /b 1
)

echo [INFO] Using model: %MODEL%
%PYEXE% -m ai.play_ppo --model "%MODEL%" --episodes 3
if errorlevel 1 (
  echo.
  echo [ERROR] Playback failed. Send me full output.
  pause
  exit /b 1
)

echo.
echo [OK] Playback finished.
pause
exit /b 0
