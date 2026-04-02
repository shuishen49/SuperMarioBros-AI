from __future__ import annotations

import retro
from stable_baselines3.common.vec_env import DummyVecEnv

from mario_rl.wrappers import (
    DiscreteRetroActions,
    EpisodicLifeRetro,
    RamGridObservation,
    ResizeObservation,
    RetroRewardWrapper,
)
from mario_rl.utils import ensure_dir


def _make_single_env(
    game: str = 'SuperMarioBros-Nes',
    state: str | None = None,
    scenario: str | None = None,
    frame_skip: int = 4,
    grayscale: bool = True,
    resize_shape=(84, 84),
    obs_mode: str = 'image',
    render_mode=None,
):
    kwargs = {'game': game}
    if state:
        kwargs['state'] = state
    if scenario:
        kwargs['scenario'] = scenario

    env = retro.make(**kwargs)
    env = DiscreteRetroActions(env)
    env = EpisodicLifeRetro(env)
    env = RetroRewardWrapper(env)

    mode = (obs_mode or 'image').lower()
    if mode == 'ram':
        env = RamGridObservation(env)
    else:
        env = ResizeObservation(env, shape=resize_shape, grayscale=grayscale)
    return env


def make_train_env(**kwargs):
    ensure_dir('models')
    ensure_dir('runs')
    return DummyVecEnv([lambda: _make_single_env(**kwargs)])


def make_eval_env(render_mode=None, **kwargs):
    defaults = {
        'game': 'SuperMarioBros-Nes',
        'state': None,
        'scenario': None,
        'frame_skip': 4,
        'grayscale': True,
        'resize_shape': (84, 84),
        'obs_mode': 'image',
        'render_mode': render_mode,
    }
    defaults.update(kwargs)
    return DummyVecEnv([lambda: _make_single_env(**defaults)])
