import argparse
import os
import cv2
import numpy as np

from mario_rl.config import TRAIN_CONFIG
from mario_rl.env import make_train_env
from mario_rl.callbacks import make_callbacks

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor


class RenderCallback(BaseCallback):
    def __init__(self, every_steps: int = 1):
        super().__init__()
        self.every_steps = max(1, int(every_steps))
        self.win_name = 'Mario Training Preview'
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def _on_step(self) -> bool:
        try:
            # Keep window responsive even when skipping draw.
            if self.n_calls % self.every_steps != 0:
                cv2.waitKey(1)
                return True

            frame = None

            # Prefer direct training observation (works for both image and RAM modes).
            new_obs = self.locals.get('new_obs', None)
            if new_obs is not None:
                arr = np.asarray(new_obs)
                if arr.ndim >= 4:
                    frame = arr[0]
                elif arr.ndim == 3:
                    frame = arr

            # Fallback to environment-rendered images.
            if frame is None:
                images = self.training_env.get_images()
                if images and images[0] is not None:
                    frame = images[0]

            if frame is None:
                cv2.waitKey(1)
                return True

            # RAM mode (possibly frame-stacked): small HxW with channels as temporal stack.
            if frame.ndim == 3 and frame.shape[0] <= 32 and frame.shape[1] <= 32:
                if frame.shape[-1] >= 1:
                    grid = frame[..., -1].astype(np.float32)  # use latest stacked frame
                else:
                    grid = frame.astype(np.float32)

                vis = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
                vis[grid <= -0.5] = (0, 0, 255)                      # enemy: red
                vis[(grid > -0.5) & (grid < 0.5)] = (20, 20, 20)     # empty: dark
                vis[(grid >= 0.5) & (grid < 1.5)] = (200, 200, 200)  # tile: gray
                vis[grid >= 1.5] = (0, 255, 0)                       # mario: green
                scale = 24
                frame_bgr = cv2.resize(vis, (vis.shape[1] * scale, vis.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
                cv2.putText(frame_bgr, 'RAM Grid (-1 enemy, 0 empty, 1 tile, 2 mario)', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif frame.ndim == 2:
                frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                frame_bgr = cv2.cvtColor(frame[..., 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

            cv2.imshow(self.win_name, frame_bgr)
            cv2.waitKey(1)
        except Exception:
            # Never block training on preview rendering.
            pass
        return True

    def _on_training_end(self) -> None:
        try:
            cv2.destroyWindow(self.win_name)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render preview window during training (slower)')
    parser.add_argument('--render-every', type=int, default=1, help='Render every N steps')
    parser.add_argument('--timesteps', type=int, default=None, help='Override total timesteps')
    parser.add_argument('--resume', type=str, default=None, help='Path to existing PPO model zip to continue training')
    parser.add_argument('--device', type=str, default=None, help='Override device, e.g. cpu/cuda/auto')
    parser.add_argument('--save-freq', type=int, default=None, help='Override checkpoint save frequency (steps)')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate, e.g. 1e-4')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = make_train_env(
        game=TRAIN_CONFIG['game'],
        state=TRAIN_CONFIG['state'],
        scenario=TRAIN_CONFIG['scenario'],
        frame_skip=TRAIN_CONFIG['frame_skip'],
        grayscale=TRAIN_CONFIG['grayscale'],
        resize_shape=TRAIN_CONFIG['resize_shape'],
        obs_mode=TRAIN_CONFIG.get('obs_mode', 'image'),
    )

    env = VecMonitor(env)
    n_stack = int(TRAIN_CONFIG.get('frame_stack', 1))
    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)

    device = args.device if args.device is not None else TRAIN_CONFIG['device']
    learning_rate = args.lr if args.lr is not None else TRAIN_CONFIG['learning_rate']

    if args.resume and os.path.exists(args.resume):
        model = PPO.load(args.resume, env=env, device=device)
        print(f'Resuming training from: {args.resume}')
        # Ensure lr override also applies when resuming from checkpoint.
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        reset_num_timesteps = False
    else:
        model = PPO(
            policy=TRAIN_CONFIG.get('policy', 'CnnPolicy'),
            env=env,
            learning_rate=learning_rate,
            n_steps=TRAIN_CONFIG['n_steps'],
            batch_size=TRAIN_CONFIG['batch_size'],
            n_epochs=TRAIN_CONFIG['n_epochs'],
            gamma=TRAIN_CONFIG['gamma'],
            gae_lambda=TRAIN_CONFIG['gae_lambda'],
            clip_range=TRAIN_CONFIG['clip_range'],
            ent_coef=TRAIN_CONFIG['ent_coef'],
            vf_coef=TRAIN_CONFIG['vf_coef'],
            verbose=1,
            tensorboard_log=TRAIN_CONFIG['tensorboard_log'],
            device=device,
        )
        reset_num_timesteps = True

    print(f'Using learning_rate={learning_rate}')

    save_dir = os.path.dirname(TRAIN_CONFIG['save_path'])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save once before learning so live demo has something to load immediately
    model.save(TRAIN_CONFIG['save_path'])

    save_freq = args.save_freq if args.save_freq is not None else TRAIN_CONFIG['save_freq']
    base_callbacks = make_callbacks(TRAIN_CONFIG['save_path'], save_freq)
    callback_items = [base_callbacks]
    if args.render:
        callback_items.append(RenderCallback(every_steps=args.render_every))
    callbacks = CallbackList(callback_items)

    total_timesteps = args.timesteps if args.timesteps is not None else TRAIN_CONFIG['total_timesteps']

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(TRAIN_CONFIG['save_path'])
    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
