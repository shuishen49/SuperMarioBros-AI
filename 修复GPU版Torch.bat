@echo off
chcp 65001 >nul
cd /d %~dp0

echo [FIX] 安装 CUDA 版 PyTorch（cu121）...
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897

D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe -m pip install --proxy http://127.0.0.1:7897 --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo.
echo [CHECK] 验证 CUDA 可用性...
D:\source_code\mario-yolo-ppo\.venv\Scripts\python.exe -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

pause
