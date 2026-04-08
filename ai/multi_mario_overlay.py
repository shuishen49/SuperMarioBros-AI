import os
import json
from typing import Any

import pygame as pg
from stable_baselines3.common.callbacks import BaseCallback

from source import constants as c
from source import tools

class MultiMarioOverlayCallback(BaseCallback):
    """Render many vector-env Mario agents in ONE shared window (not one window per env)."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, max_agents: int = 2, use_raw_frames: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.width = width
        self.height = height
        self.fps = fps
        self.max_agents = max_agents
        self.use_raw_frames = use_raw_frames

        self.screen = None
        self.clock = None
        self.font = None

        self.bg = None
        self.map_width = 3400
        self.map_height = c.SCREEN_HEIGHT

        self.mario_image = None
        self.mario_scale = 1.0
        self.mario_sheet = None
        self.frame_rects_by_form = {
            "small": [],
            "big": [],
            "fire": [],
        }
        self.sprite_cache = {}

    def _load_background(self):
        try:
            bg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "graphics", "level_1.png")
            raw = pg.image.load(bg_path).convert()
            scaled = pg.transform.scale(
                raw,
                (
                    int(raw.get_width() * c.BACKGROUND_MULTIPLER),
                    int(raw.get_height() * c.BACKGROUND_MULTIPLER),
                ),
            )
            self.bg = scaled
            self.map_width = max(self.map_width, self.bg.get_width())
            self.map_height = self.bg.get_height()
        except Exception:
            self.bg = None

    def _load_mario_sprite(self):
        try:
            sheet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "graphics", "mario_bros.png")
            self.mario_sheet = pg.image.load(sheet_path).convert()

            player_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "source", "data", "player", "mario.json")
            with open(player_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            image_frames = data["image_frames"]
            self.frame_rects_by_form = {
                "small": [
                    pg.Rect(int(fr["x"]), int(fr["y"]), int(fr["width"]), int(fr["height"]))
                    for fr in image_frames.get("right_small_normal", [])
                ],
                "big": [
                    pg.Rect(int(fr["x"]), int(fr["y"]), int(fr["width"]), int(fr["height"]))
                    for fr in image_frames.get("right_big_normal", [])
                ],
                "fire": [
                    pg.Rect(int(fr["x"]), int(fr["y"]), int(fr["width"]), int(fr["height"]))
                    for fr in image_frames.get("right_big_fire", [])
                ],
            }

            first_list = self.frame_rects_by_form["small"] or self.frame_rects_by_form["big"] or self.frame_rects_by_form["fire"]
            first = first_list[0]
            self.mario_image = tools.get_image(
                self.mario_sheet,
                first.x,
                first.y,
                first.w,
                first.h,
                c.BLACK,
                c.SIZE_MULTIPLIER * self.mario_scale,
            )
            self.sprite_cache = {}
        except Exception:
            self.mario_image = None
            self.mario_sheet = None
            self.frame_rects_by_form = {
                "small": [],
                "big": [],
                "fire": [],
            }

    def _on_training_start(self) -> None:
        if not pg.get_init():
            pg.init()
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("PPO Same-Screen Multi-Mario Monitor")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("consolas", 20)
        self._load_background()
        self._load_mario_sprite()

    def _safe_num(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _on_step(self) -> bool:
        if self.screen is None:
            return True

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False

        infos = self.locals.get("infos", []) or []

        positions = []
        primary = infos[0] if infos and isinstance(infos[0], dict) else {}
        # Shared camera anchor:
        # 优先跟随“第一个还活着的马里奥”，避免 env-0 死后视角锁死在出生点。
        camera_owner = primary
        for info in infos:
            if isinstance(info, dict) and not bool(info.get("player_dead", False)):
                camera_owner = info
                break
        camera_x = int(self._safe_num(camera_owner.get("viewport_x", 0.0), 0.0))

        for i, info in enumerate(infos[: max(1, self.max_agents)]):
            if not isinstance(info, dict):
                continue
            dead = bool(info.get("player_dead", False))

            # IMPORTANT:
            # - info['screen_x'] is relative to EACH env's own camera.
            # - In same-screen overlay we must project ALL agents with ONE shared camera (env-0).
            # Otherwise, after respawn/camera jumps, clones appear at wrong places (often spawn area).
            world_x = self._safe_num(info.get("world_x", 0.0), 0.0)
            sx = world_x - camera_x

            # Vertical camera in this project is effectively fixed; keep using env-provided screen_bottom.
            bottom = self._safe_num(info.get("screen_bottom", c.GROUND_HEIGHT), c.GROUND_HEIGHT)
            frame_idx = int(self._safe_num(info.get("frame_index", 0), 0))
            facing_right = bool(info.get("facing_right", True))
            is_big = bool(info.get("player_big", False))
            is_fire = bool(info.get("player_fire", False))
            positions.append((i, sx, bottom, dead, frame_idx, facing_right, world_x, is_big, is_fire))

        self.screen.fill((20, 22, 32))

        if self.use_raw_frames and hasattr(self.training_env, "env_method"):
            try:
                frames = self.training_env.env_method("get_render_frame")
                if frames:
                    base = frames[0]
                    h, w = base.shape[0], base.shape[1]
                    world = pg.surfarray.make_surface(base.swapaxes(0, 1))
                    world = world.convert()
                    if self.width != w or self.height != h:
                        world = pg.transform.scale(world, (self.width, self.height))
                    self.screen.blit(world, (0, 0))
            except Exception:
                self.use_raw_frames = False

        if not self.use_raw_frames:
            if self.bg is not None:
                y_offset = max(0, self.height - self.bg.get_height())
                self.screen.blit(self.bg, (-camera_x, y_offset))
            else:
                pg.draw.rect(self.screen, (45, 130, 220), (0, 0, self.width, self.height))
                pg.draw.rect(self.screen, (90, 180, 75), (0, self.height - 90, self.width, 90))

        # draw many marios in one shared viewport (aligned with env-0 camera)
        for idx, sx, bottom, dead, frame_idx, facing_right, world_x, is_big, is_fire in positions:
            sx = int(sx)
            bottom = int(bottom)
            if 0 <= sx < self.width:
                if self.use_raw_frames and idx == 0:
                    # 仅在使用真实底图时跳过 env-0，避免重复覆盖
                    continue

                if self.mario_sheet is not None:
                    form = "fire" if is_fire else ("big" if is_big else "small")
                    frame_rects = self.frame_rects_by_form.get(form, [])
                    if not frame_rects:
                        frame_rects = self.frame_rects_by_form.get("small", [])
                    if not frame_rects:
                        continue

                    rect = frame_rects[frame_idx % len(frame_rects)]
                    cache_key = (form, frame_idx % len(frame_rects), facing_right)
                    img = self.sprite_cache.get(cache_key)
                    if img is None:
                        img = tools.get_image(
                            self.mario_sheet,
                            rect.x,
                            rect.y,
                            rect.w,
                            rect.h,
                            c.BLACK,
                            c.SIZE_MULTIPLIER,
                        )
                        if not facing_right:
                            img = pg.transform.flip(img, True, False)
                        self.sprite_cache[cache_key] = img

                    draw_x = sx - img.get_width() // 2
                    draw_y = bottom - img.get_height()
                    self.screen.blit(img, (draw_x, draw_y))
                    if dead:
                        pg.draw.line(self.screen, (255, 80, 80), (draw_x, draw_y), (draw_x + img.get_width(), draw_y + img.get_height()), 2)

        camera_owner_idx = "?"
        for i, info in enumerate(infos):
            if info is camera_owner:
                camera_owner_idx = str(i)
                break

        info_text = (
            f"envs(shown)={len(positions)} (overlay clones={max(0, len(positions)-1)})   "
            f"timesteps={self.num_timesteps}   camera_x={camera_x}   cam_env={camera_owner_idx}"
        )
        txt = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(txt, (16, 12))

        pg.display.flip()
        self.clock.tick(self.fps)
        return True

    def _on_training_end(self) -> None:
        if self.screen is not None:
            try:
                pg.display.quit()
            except Exception:
                pass
            self.screen = None
