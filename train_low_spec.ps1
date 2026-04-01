$env:PYTHONUNBUFFERED='1'
Write-Host 'Starting low-spec Mario PPO training...' -ForegroundColor Cyan
python check_device.py
python train_ppo.py
