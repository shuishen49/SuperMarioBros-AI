import os
import argparse

import pygame as pg
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from ai.multi_mario_overlay import MultiMarioOverlayCallback

from ai.ppo_env import MarioPPOEnv


def make_env(level_num=1, frame_skip=4, max_steps=4500, render_mode="none", stuck_frames=360):
    def _init():
        return MarioPPOEnv(
            level_num=level_num,
            frame_skip=frame_skip,
            max_steps=max_steps,
            render_mode=render_mode,
            stuck_frames=stuck_frames,
        )

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on PythonSuperMario")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--model-name", type=str, default="ppo_mario")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--render-mode", type=str, default="none", choices=["none", "human"])
    parser.add_argument("--overlay", action="store_true", help="单窗口同屏监视多环境（推荐配合 render-mode=none）")
    parser.add_argument("--overlay-width", type=int, default=800)
    parser.add_argument("--overlay-height", type=int, default=600)
    parser.add_argument("--overlay-fps", type=int, default=30)
    parser.add_argument("--overlay-agents", type=int, default=2, help="同屏展示总人数（含第1个真实马里奥）")
    parser.add_argument("--overlay-no-raw", action="store_true", help="禁用真实底图采样，纯精灵同屏渲染（更稳）")
    parser.add_argument("--stuck-frames", type=int, default=360, help="X轴连续无有效前进多少帧判定为死亡/结束")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.render_mode == "human" and args.n_envs != 1:
        print("[INFO] render_mode=human 时自动将 --n-envs 设为 1（可视化训练更稳定）")
        args.n_envs = 1

    if args.overlay and args.render_mode == "human":
        print("[INFO] --overlay 与 render-mode=human 不建议同用，自动改为 render-mode=none")
        args.render_mode = "none"

    env_fns = [
        make_env(args.level, args.frame_skip, args.max_steps, args.render_mode, args.stuck_frames)
        for _ in range(args.n_envs)
    ]
    if args.vec_env == "subproc":
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=args.stack, channels_order="last")

    if args.resume and os.path.exists(args.resume):
        model = PPO.load(args.resume, env=vec_env, device="auto")
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=args.log_dir,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            learning_rate=2.5e-4,
            device="auto",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=args.save_dir,
        name_prefix=args.model_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = [checkpoint_cb]
    if args.overlay:
        callbacks.append(
            MultiMarioOverlayCallback(
                width=args.overlay_width,
                height=args.overlay_height,
                fps=args.overlay_fps,
                max_agents=args.overlay_agents,
                use_raw_frames=not args.overlay_no_raw,
            )
        )

    model.learn(total_timesteps=args.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)

    final_path = os.path.join(args.save_dir, f"{args.model_name}_final")
    model.save(final_path)
    vec_env.close()
    pg.quit()
    print(f"Saved final model to: {final_path}.zip")


if __name__ == "__main__":
    main()
