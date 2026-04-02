import argparse
import glob
import os
import time
from collections import deque

import cv2
import numpy as np
import retro
from stable_baselines3 import PPO

from mario_rl.config import TRAIN_CONFIG
from mario_rl.wrappers import DiscreteRetroActions, EpisodicLifeRetro


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='auto', help='auto | explicit model path')
    p.add_argument('--episode-steps', type=int, default=4000)
    p.add_argument('--reload-seconds', type=float, default=2.0)
    p.add_argument('--show-every', type=int, default=4)
    p.add_argument('--scale', type=int, default=3, help='Display scale for color preview window')
    p.add_argument('--device', type=str, default='cpu', help='Device for demo model inference: cpu/cuda/auto')
    return p.parse_args()


def preprocess_frame(rgb_frame: np.ndarray, out_size=(84, 84)) -> np.ndarray:
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.uint8)


def get_ram(env):
    u = env.unwrapped
    if hasattr(u, 'get_ram'):
        return u.get_ram()
    if hasattr(u, 'ram'):
        return u.ram
    return None


def tile_loc_to_ram_address(x, y):
    page = x // 16
    x_loc = x % 16
    y_loc = page * 13 + y
    return 0x500 + x_loc + y_loc * 16


def build_ram_grid(env) -> np.ndarray:
    ram = get_ram(env)
    if ram is None:
        return np.zeros((13, 16), dtype=np.float32)

    mario_level_x = int(ram[0x6D]) * 256 + int(ram[0x86])
    mario_x = int(ram[0x3AD])
    mario_y = int(ram[0x3B8]) + 16
    x_start = mario_level_x - mario_x

    grid = np.zeros((13, 16), dtype=np.float32)
    screen_start = int(np.rint(x_start / 16.0))

    for i in range(16):
        for j in range(13):
            x_loc = (screen_start + i) % 32
            addr = tile_loc_to_ram_address(x_loc, j)
            if int(ram[addr]) != 0:
                grid[j, i] = 1.0

    mx = (mario_x + 8) // 16
    my = (mario_y - 32) // 16
    if 0 <= mx < 16 and 0 <= my < 13:
        grid[my, mx] = 2.0

    for i in range(5):
        if int(ram[0x0F + i]) == 1:
            enemy_x = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - x_start
            enemy_y = int(ram[0xCF + i])
            ex = (enemy_x + 8) // 16
            ey = (enemy_y + 8 - 32) // 16
            if 0 <= ex < 16 and 0 <= ey < 13:
                grid[ey, ex] = -1.0

    return grid


def make_model_obs(frame_stack: deque, mode: str) -> np.ndarray:
    if mode == 'ram':
        arr = np.stack(list(frame_stack), axis=-1).astype(np.float32)  # H,W,C
        return np.expand_dims(arr, axis=0)
    arr = np.stack(list(frame_stack), axis=0)  # C,H,W
    return np.expand_dims(arr, axis=0)


def latest_checkpoint_path(models_dir='models'):
    pattern = os.path.join(models_dir, 'ppo_mario_checkpoint_*_steps.zip')
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def resolve_model_path(model_arg: str):
    if model_arg and model_arg.lower() != 'auto':
        return model_arg

    # Prefer latest checkpoint.
    latest_ckpt = latest_checkpoint_path('models')
    if latest_ckpt:
        return latest_ckpt

    # Fallbacks for both old image model and new RAM model.
    cands = [
        os.path.join('models', 'best', 'ppo_mario_ram.zip'),
        os.path.join('models', 'best', 'ppo_mario_retro.zip'),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def main():
    args = parse_args()

    env = retro.make(game=TRAIN_CONFIG['game'])
    env = DiscreteRetroActions(env)
    env = EpisodicLifeRetro(env)
    cv2.namedWindow('Mario Live Demo (best model, color)', cv2.WINDOW_NORMAL)

    last_load = 0.0
    model = None
    current_model_path = None
    obs_mode = 'image'
    stack_len = int(TRAIN_CONFIG.get('frame_stack', 4))

    try:
        while True:
            now = time.time()
            should_reload = (model is None) or ((now - last_load) >= args.reload_seconds)
            if should_reload:
                path = resolve_model_path(args.model)
                if path is None:
                    frame = np.zeros((360, 900, 3), dtype=np.uint8)
                    cv2.putText(frame, 'Waiting for model file...', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.imshow('Mario Live Demo (best model, color)', frame)
                    if (cv2.waitKey(300) & 0xFF) == 27:
                        break
                    continue

                if model is None or path != current_model_path or (now - last_load) >= args.reload_seconds:
                    try:
                        model = PPO.load(path, device=args.device)
                        current_model_path = path
                        last_load = now

                        shape = tuple(model.observation_space.shape)
                        # RAM model expected shape: (13, 16, n_stack)
                        if len(shape) == 3 and shape[0] == 13 and shape[1] == 16:
                            obs_mode = 'ram'
                            stack_len = int(shape[2])
                        else:
                            obs_mode = 'image'
                            stack_len = int(TRAIN_CONFIG.get('frame_stack', 4))

                        print(f'[demo] loaded model: {path}')
                        print(f'[demo] obs_mode={obs_mode}, model_obs_shape={shape}')
                    except Exception as e:
                        frame = np.zeros((360, 900, 3), dtype=np.uint8)
                        cv2.putText(frame, 'Waiting for model...', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(frame, str(e)[:100], (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                        cv2.imshow('Mario Live Demo (best model, color)', frame)
                        if (cv2.waitKey(300) & 0xFF) == 27:
                            break
                        continue

            obs = env.reset()  # RGB frame
            if obs_mode == 'ram':
                first = build_ram_grid(env)
            else:
                first = preprocess_frame(obs, out_size=TRAIN_CONFIG['resize_shape'])
            frame_stack = deque([first.copy() for _ in range(stack_len)], maxlen=stack_len)

            total_reward = 0.0

            for step in range(args.episode_steps):
                model_obs = make_model_obs(frame_stack, mode=obs_mode)
                action, _ = model.predict(model_obs, deterministic=True)
                env_action = action[0] if hasattr(action, '__len__') else action

                obs, reward, done, info = env.step(env_action)
                total_reward += float(reward)

                if obs_mode == 'ram':
                    frame_stack.append(build_ram_grid(env))
                else:
                    frame_stack.append(preprocess_frame(obs, out_size=TRAIN_CONFIG['resize_shape']))

                if step % max(1, args.show_every) == 0:
                    frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                    if args.scale > 1:
                        frame_bgr = cv2.resize(
                            frame_bgr,
                            (frame_bgr.shape[1] * args.scale, frame_bgr.shape[0] * args.scale),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    cv2.putText(frame_bgr, f'demo reward: {total_reward:.1f}', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    base = os.path.basename(current_model_path) if current_model_path else 'None'
                    cv2.putText(frame_bgr, f'model: {base}', (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame_bgr, f'obs_mode: {obs_mode}', (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    cv2.imshow('Mario Live Demo (best model, color)', frame_bgr)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return

                if done:
                    break

            print(f'[demo] episode reward={total_reward:.2f}')

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
