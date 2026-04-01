import argparse

from mario_rl.config import TRAIN_CONFIG
from mario_rl.env import make_eval_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained PPO model zip')
    parser.add_argument('--render', action='store_true', help='Render game window')
    parser.add_argument('--episodes', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_eval_env(
        game=TRAIN_CONFIG['game'],
        state=TRAIN_CONFIG['state'],
        scenario=TRAIN_CONFIG['scenario'],
        render_mode='human' if args.render else None,
    )
    env = VecFrameStack(env, n_stack=TRAIN_CONFIG['frame_stack'])

    model = PPO.load(args.model)

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_reward += float(reward[0])
            done = bool(dones[0])

        print(f'Episode {episode + 1}: total_reward={total_reward:.2f}')

    env.close()


if __name__ == '__main__':
    main()
