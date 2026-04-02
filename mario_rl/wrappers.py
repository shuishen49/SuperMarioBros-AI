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


class RamGridObservation(gym.ObservationWrapper):
    """Convert retro RAM to a compact 13x16 semantic grid.

    Values:
      -1 enemy
       0 empty
       1 solid tile/item
       2 Mario
    """

    def __init__(self, env):
        super().__init__(env)
        self.screen_size_x = 16
        self.screen_size_y = 13
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(self.screen_size_y, self.screen_size_x, 1),
            dtype=np.float32,
        )

    def observation(self, obs):
        ram = self._get_ram()
        if ram is None:
            return np.zeros((self.screen_size_y, self.screen_size_x, 1), dtype=np.float32)

        mario_level_x = int(ram[0x6D]) * 256 + int(ram[0x86])
        mario_x = int(ram[0x3AD])
        mario_y = int(ram[0x3B8]) + 16
        x_start = mario_level_x - mario_x

        grid = np.zeros((self.screen_size_y, self.screen_size_x), dtype=np.float32)

        screen_start = int(np.rint(x_start / 16.0))
        for i in range(self.screen_size_x):
            for j in range(self.screen_size_y):
                x_loc = (screen_start + i) % (self.screen_size_x * 2)
                addr = self._tile_loc_to_ram_address(x_loc, j)
                if int(ram[addr]) != 0:
                    grid[j, i] = 1.0

        # Mario
        mx = (mario_x + 8) // 16
        my = (mario_y - 32) // 16
        if 0 <= mx < self.screen_size_x and 0 <= my < self.screen_size_y:
            grid[my, mx] = 2.0

        # Enemies
        for i in range(5):
            if int(ram[0x0F + i]) == 1:
                enemy_x = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - x_start
                enemy_y = int(ram[0xCF + i])
                ex = (enemy_x + 8) // 16
                ey = (enemy_y + 8 - 32) // 16
                if 0 <= ex < self.screen_size_x and 0 <= ey < self.screen_size_y:
                    grid[ey, ex] = -1.0

        return np.expand_dims(grid, axis=-1).astype(np.float32)

    def _get_ram(self):
        u = self.env.unwrapped
        try:
            if hasattr(u, 'get_ram'):
                return u.get_ram()
            if hasattr(u, 'ram'):
                return u.ram
        except Exception:
            return None
        return None

    @staticmethod
    def _tile_loc_to_ram_address(x, y):
        page = x // 16
        x_loc = x % 16
        y_loc = page * 13 + y
        return 0x500 + x_loc + y_loc * 16


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
    def __init__(self, env, stuck_patience: int = 180, stuck_penalty: float = -5.0):
        super().__init__(env)
        self.prev_x = None
        self.prev_score = 0
        self.stuck_steps = 0
        self.stuck_patience = int(stuck_patience)
        self.stuck_penalty = float(stuck_penalty)

    def reset(self, **kwargs):
        self.prev_x = None
        self.prev_score = 0
        self.stuck_steps = 0
        return self.env.reset(**kwargs)

    @staticmethod
    def _get_x_pos(info: dict):
        for k in ('x_pos', 'x', 'screen_x_pos', 'xscrollLo'):
            if k in info and info[k] is not None:
                try:
                    return float(info[k])
                except Exception:
                    pass
        return None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        info = info if isinstance(info, dict) else {}
        x_pos = self._get_x_pos(info)
        score = info.get('score', 0)

        if x_pos is None:
            x_pos = self.prev_x if self.prev_x is not None else 0.0

        prev_x = self.prev_x if self.prev_x is not None else x_pos
        delta_x = float(x_pos) - float(prev_x)

        progress_reward = max(0.0, delta_x) * 0.05
        score_reward = max(0.0, float(score) - float(self.prev_score)) * 0.01
        shaped_reward = float(reward) + progress_reward + score_reward

        # If Mario stays in place too long, terminate episode and restart.
        if delta_x <= 0.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if (not done) and self.stuck_steps >= self.stuck_patience:
            done = True
            shaped_reward += self.stuck_penalty
            info['stuck_done'] = True
            info['stuck_steps'] = self.stuck_steps

        self.prev_x = float(x_pos)
        self.prev_score = float(score)
        return obs, shaped_reward, done, info
