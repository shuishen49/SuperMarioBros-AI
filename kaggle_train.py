import os
import sys
import glob
import subprocess


def run(cmd):
    print('>>>', cmd)
    return subprocess.run(cmd, shell=True, check=False)


def find_rom_zip():
    # 1) 优先环境变量
    env_path = os.environ.get('ROM_ZIP_PATH', '').strip()
    if env_path and os.path.exists(env_path):
        return env_path

    # 2) 常见默认路径
    candidates = [
        '/kaggle/input/smb-rom/Super Mario Bros. (World).zip',
        '/kaggle/input/smb-rom/SuperMarioBros.zip',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 3) 全量扫描 /kaggle/input 下 zip
    zips = glob.glob('/kaggle/input/**/*.zip', recursive=True)
    if not zips:
        return None

    # 优先包含关键词的 zip
    keys = ('mario', 'super', 'bros', 'nes')
    scored = []
    for p in zips:
        low = p.lower()
        score = sum(k in low for k in keys)
        scored.append((score, p))
    scored.sort(reverse=True)
    return scored[0][1]


# 0) 环境检查
run('nvidia-smi')
run(f'{sys.executable} -V')

# 1) 拉 main 分支仓库
WORKDIR = '/kaggle/working'
REPO_DIR = os.path.join(WORKDIR, 'SuperMarioBros-AI')
if not os.path.exists(REPO_DIR):
    run(f'git clone -b main https://github.com/shuishen49/SuperMarioBros-AI.git {REPO_DIR}')
os.chdir(REPO_DIR)
run('ls -la')

# 2) 安装依赖
run(f'{sys.executable} -m pip install -U pip')
run(f'{sys.executable} -m pip install -r requirements.txt')
run(f'{sys.executable} -m pip install "gym==0.23.1" "shimmy>=2.0"')

# 3) 导入 ROM（自动查找）
rom_zip_path = find_rom_zip()
assert rom_zip_path and os.path.exists(rom_zip_path), (
    'ROM 文件不存在。请在 Kaggle 右侧 Add Data 上传包含 SMB ROM 的数据集，'
    '或设置环境变量 ROM_ZIP_PATH。'
)
print('[ROM] Using:', rom_zip_path)
run('python -m retro.import "{}"'.format(rom_zip_path))

# 4) 开始训练（无渲染）
TIMESTEPS = int(os.environ.get('TIMESTEPS', '100000'))
run(f'{sys.executable} train_ppo.py --timesteps {TIMESTEPS} --device cuda')

# 5) 导出模型
run('mkdir -p /kaggle/working/output_models')
run('cp -r models/* /kaggle/working/output_models/ || true')
run('ls -lah /kaggle/working/output_models')

print('\nDone. 模型在 /kaggle/working/output_models')
