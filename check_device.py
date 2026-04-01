import torch

print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu name:', torch.cuda.get_device_name(0))
    print('gpu count:', torch.cuda.device_count())
else:
    print('No CUDA GPU detected, training will run on CPU and be much slower.')
