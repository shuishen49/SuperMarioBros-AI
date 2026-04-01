import cv2
import gym
import numpy as np


class DiscreteRetroActions(gym.ActionWrapper):
    """Map a small discrete action set to Retro's multi-binary button array."""

    def __init__(self, env, action_names=None):
        super().__init__(env)
        self.buttons = list(getattr(env.unwrapped, 'buttons', []))
        # Keep a compact set that encourages moving right + jumping.
        self.action_names = action_names or [
            ['RIGHT', 'B'],
            ['RIGHT', 'A', 'B'],
            ['DOWN'],
        ]
        self.action_space = gym.spaces.Discrete(len(self.action_names))

    def action(self, act):
        idx = int(act)
        pressed = set(self.action_names[idx])
        arr = np.zeros(len(self.buttons), dtype=np.int8)
        for i, btn in enumerate(self.buttons):
            if btn in pressed:
                arr[i] = 1
        return arr


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84), grayscale=True):
        super().__init__(env)
        self.shape = shape
        self.grayscale = grayscale

        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], channels),
            dtype=np.uint8,
        )

    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = np.expand_dims(obs, axis=-1)
        return obs.astype(np.uint8)


class EpisodicLifeRetro(gym.Wrapper):
    """Treat life loss as end of episode (do not wait until all lives are gone)."""

    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = None

    @staticmethod
    def _get_lives(info):
        for k in ('lives', 'life', 'player_lives'):
            if k in info and info[k] is not None:
                try:
                    return int(info[k])
                except Exception:
                    pass
        return None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_lives = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        lives = self._get_lives(info if isinstance(info, dict) else {})
        life_lost = False
        if lives is not None and self.prev_lives is not None and lives < self.prev_lives:
            life_lost = True

        self.prev_lives = lives if lives is not None else self.prev_lives

        if life_lost and not done:
            done = True
            if isinstance(info, dict):
                info['life_lost_done'] = True

        return obs, reward, done, info


class RetroRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_x = 0
        self.prev_score = 0

    def reset(self, **kwargs):
        self.prev_x = 0
        self.prev_score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        x_pos = info.get('x', info.get('xscrollLo', 0))
        score = info.get('score', 0)

        progress_reward = max(0, x_pos - self.prev_x) * 0.05
        score_reward = max(0, score - self.prev_score) * 0.01
        shaped_reward = float(reward) + progress_reward + score_reward

        self.prev_x = x_pos
        self.prev_score = score
        return obs, shaped_reward, done, info
