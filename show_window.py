import argparse
import time
import cv2
import retro


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--realtime', action='store_true', help='Cap to ~60 FPS for human viewing')
    parser.add_argument('--show-every', type=int, default=1, help='Display every N env steps (higher = faster)')
    return parser.parse_args()


def main():
    args = parse_args()

    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()
    cv2.namedWindow('Mario Preview', cv2.WINDOW_NORMAL)

    step = 0
    t0 = time.time()

    while step < args.max_steps:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1

        if step % max(1, args.show_every) == 0:
            frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imshow('Mario Preview', frame_bgr)
            delay = 16 if args.realtime else 1
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:
                break

        if done:
            obs = env.reset()

    dt = max(1e-6, time.time() - t0)
    print(f'Ran {step} steps in {dt:.2f}s, approx {step/dt:.1f} steps/s')

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
