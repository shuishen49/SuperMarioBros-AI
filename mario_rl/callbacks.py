from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


def make_callbacks(save_path: str, save_freq: int):
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='models/',
        name_prefix='ppo_mario_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    return CallbackList([checkpoint_callback])
