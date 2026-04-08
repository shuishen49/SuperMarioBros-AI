import pygame as pg
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from source import constants as c
from source import tools
from source import setup
from source.states.level import Level


class _KeyState:
    """Minimal key-state object compatible with existing player code."""

    def __init__(self):
        self._pressed = set()

    def set_pressed(self, keys):
        self._pressed = set(keys)

    def __getitem__(self, key):
        return key in self._pressed


class MarioPPOEnv(gym.Env):
    """Gymnasium env wrapper for this pygame Super Mario project."""

    metadata = {"render_modes": ["none", "human"], "render_fps": 60}

    ACTIONS = {
        0: (),
        1: (tools.keybinding['right'],),
        2: (tools.keybinding['right'], tools.keybinding['jump']),
        3: (tools.keybinding['jump'],),
        4: (tools.keybinding['left'],),
        5: (tools.keybinding['right'], tools.keybinding['action']),
        6: (tools.keybinding['right'], tools.keybinding['action'], tools.keybinding['jump']),
    }

    def __init__(self, level_num=1, frame_skip=4, max_steps=4500, obs_size=84, render_mode="none", stuck_frames=360):
        super().__init__()
        self.level_num = level_num
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.obs_size = obs_size
        self.render_mode = render_mode
        self.stuck_frames = max(60, int(stuck_frames))

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_size, self.obs_size, 1),
            dtype=np.uint8,
        )

        self.clock = pg.time.Clock()
        self.keys = _KeyState()
        self.current_time = 0
        self.step_count = 0

        self.level = None
        self.game_info = None
        self.prev_world_x = 0.0
        self.prev_score = 0
        self.no_progress_frames = 0

        # Headless training uses an off-screen surface; human mode draws to window.
        self.surface = setup.SCREEN if self.render_mode == "human" else pg.Surface(c.SCREEN_SIZE)

    def _new_game_info(self):
        return {
            c.COIN_TOTAL: 0,
            c.SCORE: 0,
            c.LIVES: 1,
            c.TOP_SCORE: 0,
            c.CURRENT_TIME: 0.0,
            c.LEVEL_NUM: self.level_num,
            c.PLAYER_NAME: c.PLAYER_MARIO,
        }

    def _world_x(self):
        # Player rect is already world-space in this project.
        return float(self.level.player.rect.centerx)

    def _get_obs(self):
        scaled = pg.transform.smoothscale(self.surface, (self.obs_size, self.obs_size))
        arr = pg.surfarray.array3d(scaled)  # (w, h, c)
        arr = np.transpose(arr, (1, 0, 2))  # (h, w, c)
        gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return gray[..., None]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.step_count = 0

        self.game_info = self._new_game_info()
        self.level = Level()
        self.level.startup(self.current_time, self.game_info)

        self.prev_world_x = self._world_x()
        self.prev_score = self.game_info[c.SCORE]
        self.no_progress_frames = 0

        self.level.draw(self.surface)
        obs = self._get_obs()
        info = {"world_x": self.prev_world_x, "score": self.prev_score}
        return obs, info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        episode_end_reason = ""

        pressed = self.ACTIONS[int(action)]

        for _ in range(self.frame_skip):
            pg.event.pump()
            self.keys.set_pressed(pressed)

            self.current_time += int(1000 / 60)
            self.level.update(self.surface, self.keys, self.current_time)

            now_x = self._world_x()
            now_score = self.game_info[c.SCORE]
            delta_x = now_x - self.prev_world_x
            delta_score = now_score - self.prev_score

            # Reward: move right + score gain, slight penalty for moving left.
            reward += (delta_x * 0.05) + (delta_score * 0.1)
            if delta_x < 0:
                reward += delta_x * 0.03

            self.prev_world_x = now_x
            self.prev_score = now_score

            if delta_x > 0.10:
                self.no_progress_frames = 0
            else:
                self.no_progress_frames += 1

            if self.level.player.dead:
                reward -= 50.0
                terminated = True
                episode_end_reason = "dead"
                break

            if self.no_progress_frames >= self.stuck_frames:
                reward -= 30.0
                terminated = True
                episode_end_reason = "stalled"
                break

            if self.level.done:
                # Level clear bonus (not dead and reached end-state)
                if not self.level.player.dead:
                    reward += 100.0
                terminated = True
                episode_end_reason = "level_done"
                break

            self.step_count += 1
            if self.step_count >= self.max_steps:
                truncated = True
                episode_end_reason = "max_steps"
                break

        if self.render_mode == "human":
            pg.display.update()
            self.clock.tick(self.metadata["render_fps"])

        obs = self._get_obs()
        viewport_x = float(getattr(self.level.viewport, "x", 0))
        screen_x = float(self.level.player.rect.centerx - viewport_x)
        screen_bottom = float(self.level.player.rect.bottom)
        info = {
            "world_x": self.prev_world_x,
            "world_y": float(self.level.player.rect.centery),
            "viewport_x": viewport_x,
            "screen_x": screen_x,
            "screen_bottom": screen_bottom,
            "frame_index": int(getattr(self.level.player, "frame_index", 0)),
            "facing_right": bool(getattr(self.level.player, "facing_right", True)),
            "player_big": bool(getattr(self.level.player, "big", False)),
            "player_fire": bool(getattr(self.level.player, "fire", False)),
            "score": self.prev_score,
            "step_count": self.step_count,
            "level_done": bool(self.level.done),
            "player_dead": bool(self.level.player.dead),
            "stalled_out": bool(episode_end_reason == "stalled"),
            "episode_end_reason": episode_end_reason,
            "map_width": float(getattr(self.level.bg_rect, "w", c.SCREEN_WIDTH)),
            "map_height": float(getattr(self.level.bg_rect, "h", c.SCREEN_HEIGHT)),
        }
        return obs, reward, terminated, truncated, info

    def get_render_frame(self):
        """Return current RGB frame (H, W, 3) for external monitor overlay."""
        self.level.draw(self.surface)
        arr = pg.surfarray.array3d(self.surface)  # (w, h, c)
        return np.transpose(arr, (1, 0, 2))       # (h, w, c)

    def render(self):
        if self.render_mode == "human":
            pg.display.update()

    def close(self):
        pg.quit()
