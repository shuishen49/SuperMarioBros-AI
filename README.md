# Mario Retro YOLO + PPO

一个用于实现 **AI 玩 FC《超级马里奥兄弟》** 的 Python 项目骨架：

- **环境**：`gym-retro`
- **强化学习**：PPO（`stable-baselines3`）
- **视觉感知**：YOLO（`ultralytics`）
- **目标**：先跑通 `gym-retro + PPO` baseline，再为 YOLO 特征融合留接口

> 现在项目已经从 `gym-super-mario-bros + nes-py` 切换到 `gym-retro` 路线，避免在 Windows 上继续被老旧依赖恶心。

---

## 1. 推荐环境

建议：

- Windows
- Python **3.8**
- 使用 venv

### 低配机器说明（适配 GTX 1650 4G + 老 CPU）

这份项目已经按偏低配置做了保守化处理：

- 训练默认走 **84x84 灰度图**
- 单线程训练
- PPO 参数已下调，减少显存和 CPU 压力
- **YOLO 不放进训练主循环**，否则你这套配置会被拖慢得很难看
- 训练时**不要开 render**，测试时再渲染

说白了：

- **PPO baseline 能做**
- **YOLO 在线融合训练，不适合你这台机器一上来就硬怼**

### 创建虚拟环境

```powershell
cd D:\source_code\mario-yolo-ppo
py -3.8 -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

---

## 3. 训练 PPO

### 最省事：一键启动（单线程）

直接双击：

```text
一键启动.bat
```

这个脚本会自动：

- 检查 Python 3.8
- 创建 `.venv`
- 安装依赖
- 检测 CUDA
- 强制单线程环境变量
- 启动训练

### 手动启动

```powershell
python train_ppo.py
```

默认会：

- 使用 `SuperMarioBros-Nes`
- 自动保存模型到 `models/ppo_mario_retro`
- TensorBoard 日志到 `runs/`

---

## 4. 运行训练好的模型

```powershell
python play.py --model models/ppo_mario_retro.zip --render
```

---

## 5. 使用 YOLO 做检测演示

```powershell
python yolo_detect.py --source sample_frame.png
```

默认使用 `yolov8n.pt`。但要注意：

- 通用 YOLO 权重 **并不懂马里奥里的砖块、敌人、金币、管道**
- 如果你要 YOLO 真有用，需要自己标注马里奥游戏画面数据并训练自定义检测模型

---

## 6. 项目结构

```text
mario-yolo-ppo/
├─ mario_rl/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ env.py
│  ├─ wrappers.py
│  ├─ callbacks.py
│  ├─ yolo_features.py
│  └─ utils.py
├─ models/
├─ runs/
├─ train_ppo.py
├─ play.py
├─ yolo_detect.py
├─ requirements.txt
└─ README.md
```

---

## 7. gym-retro 路线的关键提醒

`gym-retro` 更适合复古游戏 RL，但它不是“安装就自带所有 ROM”。

你通常还需要：

- 对应游戏 ROM
- 可用的 integration / scenario / state

也就是说：

- 技术路线对了
- 但游戏资源和集成配置仍然要补齐

---

## 8. YOLO + PPO 的正确打开方式

### 路线 A：直接像素输入 + PPO

- 输入：游戏画面
- 模型：CNN policy
- 优点：实现最简单
- 缺点：训练效率未必高

### 路线 B：YOLO 检测结果 + PPO

- 先用 YOLO 检测：主角、敌人、金币、坑、管道、砖块等
- 再把检测框、类别、距离等编码成状态特征
- PPO 根据这些结构化特征决策

你这台机器更适合：

1. 先跑 PPO baseline
2. 再做低频 YOLO 特征提取
3. 最后再做真正融合训练

---

## 9. 当前最值得做的事

1. 先确认 `gym-retro` 能装好
2. 准备可用的 Mario ROM / integration
3. 先训练一个纯 PPO baseline
4. 后面再训练 Mario 专用 YOLO
5. 最后再做 YOLO + PPO 融合

---

## 10. 免责声明

这个项目是研究/学习用途的工程骨架，不是现成的“开箱即满级通关 AI”。别指望我写两页脚本它就自己成神。
