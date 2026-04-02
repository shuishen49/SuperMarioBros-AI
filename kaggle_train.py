import os
import glob
import shutil
import subprocess


def run(cmd, check=False):
    print('>>>', cmd)
    return subprocess.run(cmd, shell=True, check=check)


def choose_python():
    # 优先 python3.8，但必须确认它能跑 pip；否则回退 python3
    py = os.environ.get('PYTHON_BIN', '').strip()
    if py:
        return py

    if shutil.which('python3.8'):
        test = subprocess.run('python3.8 -m pip --version', shell=True, check=False)
        if test.returncode == 0:
            return 'python3.8'
        print('[WARN] python3.8 存在但 pip 不可用，回退到 python3')

    print('[WARN] 使用 python3（Kaggle 默认环境）')
    return 'python3'


def find_rom_zip():
    env_path = os.environ.get('ROM_ZIP_PATH', '').strip()
    if env_path and os.path.exists(env_path):
        return env_path

    # 常见位置：Kaggle Input + 当前仓库目录
    candidates = [
        '/kaggle/input/smb-rom/Super Mario Bros. (World).zip',
        '/kaggle/input/smb-rom/SuperMarioBros.zip',
        os.path.join(os.getcwd(), 'Super Mario Bros. (World).zip'),
        os.path.join(os.getcwd(), 'SuperMarioBros.zip'),
        os.path.join(os.getcwd(), 'rom', 'Super Mario Bros. (World).zip'),
        os.path.join(os.getcwd(), 'rom', 'SuperMarioBros.zip'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 全量扫描：input + 当前目录
    zips = []
    zips.extend(glob.glob('/kaggle/input/**/*.zip', recursive=True))
    zips.extend(glob.glob(os.path.join(os.getcwd(), '**', '*.zip'), recursive=True))
    if not zips:
        return None

    keys = ('mario', 'super', 'bros', 'nes')
    scored = []
    for p in zips:
        low = p.lower()
        score = sum(k in low for k in keys)
        scored.append((score, p))
    scored.sort(reverse=True)
    return scored[0][1]


PY = choose_python()

# 0) 环境检查
run('nvidia-smi')
run(f'{PY} -V')

# 1) 拉 main 分支仓库
WORKDIR = '/kaggle/working'
REPO_DIR = os.path.join(WORKDIR, 'SuperMarioBros-AI')
if not os.path.exists(REPO_DIR):
    run(f'git clone -b main https://github.com/shuishen49/SuperMarioBros-AI.git {REPO_DIR}', check=True)
os.chdir(REPO_DIR)
run('ls -la')

# 2) 安装依赖（先修复 gym==0.21.0 的老构建链问题）
run(f'{PY} -m pip install -U pip', check=True)
run(f'{PY} -m pip install "setuptools==65.5.0" "wheel<0.40.0" --force-reinstall', check=False)
run(f'{PY} -m pip install -r requirements.txt', check=False)
# 兜底：如果 requirements 安装里 gym 仍失败，最后再补一次关键包
run(f'{PY} -m pip install "gym==0.21.0" "shimmy>=2.0"', check=False)

# 3) 检查 retro 是否可用，再导入 ROM（自动查找）
retro_check = run(f"{PY} -c \"import retro; print('retro_ok')\"", check=False)
assert retro_check.returncode == 0, (
    '当前 Python 环境无法 import retro（gym-retro 很可能未安装成功）。'
    '如果是 Kaggle 默认 python3.12，常见为不兼容导致安装失败。'
)

rom_zip_path = find_rom_zip()
assert rom_zip_path and os.path.exists(rom_zip_path), (
    'ROM 文件不存在。请在 Kaggle 右侧 Add Data 上传包含 SMB ROM 的数据集，'
    '或设置环境变量 ROM_ZIP_PATH。'
)
print('[ROM] Using:', rom_zip_path)
import_result = run(f'{PY} -m retro.import "{rom_zip_path}"', check=False)
if import_result.returncode != 0:
    print('[WARN] 直接导入 zip 失败，尝试先解压再导入 .nes ...')
    extract_dir = '/kaggle/working/rom_extract'
    run(f'rm -rf "{extract_dir}" && mkdir -p "{extract_dir}"')
    run(f'unzip -o "{rom_zip_path}" -d "{extract_dir}"', check=False)
    nes_list = glob.glob(os.path.join(extract_dir, '**', '*.nes'), recursive=True)
    assert nes_list, 'zip 解压后未找到 .nes 文件，请检查 ROM 压缩包内容'
    nes_path = nes_list[0]
    print('[ROM] Fallback NES:', nes_path)
    run(f'{PY} -m retro.import "{nes_path}"', check=True)

# 4) 开始训练（无渲染）
TIMESTEPS = int(os.environ.get('TIMESTEPS', '100000'))
run(f'{PY} train_ppo.py --timesteps {TIMESTEPS} --device cuda', check=False)

# 5) 导出模型
run('mkdir -p /kaggle/working/output_models')
run('cp -r models/* /kaggle/working/output_models/ || true')
run('ls -lah /kaggle/working/output_models')

print('\nDone. 模型在 /kaggle/working/output_models')
