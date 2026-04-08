import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from ai.ppo_env import MarioPPOEnv


def make_env(level_num=1, frame_skip=2, max_steps=4500):
    def _init():
        return MarioPPOEnv(level_num=level_num, frame_skip=frame_skip, max_steps=max_steps, render_mode="human")

    return _init


def main():
    parser = argparse.ArgumentParser(description="Play PPO model on PythonSuperMario")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--stack", type=int, default=4)
    args = parser.parse_args()

    vec_env = DummyVecEnv([make_env(args.level, args.frame_skip, args.max_steps)])
    vec_env = VecFrameStack(vec_env, n_stack=args.stack, channels_order="last")

    model = PPO.load(args.model, env=vec_env, device="auto")

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = [False]
        total_reward = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += float(reward[0])
        print(f"Episode {ep + 1}: reward={total_reward:.2f}")

    vec_env.close()


if __name__ == "__main__":
    main()
