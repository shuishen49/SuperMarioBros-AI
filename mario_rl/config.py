TRAIN_CONFIG = {
    'game': 'SuperMarioBros-Nes',
    'state': None,
    'scenario': None,
    'frame_skip': 4,
    'grayscale': True,
    'resize_shape': (84, 84),
    'obs_mode': 'ram',  # 'ram' | 'image'
    'frame_stack': 4,
    'policy': 'MlpPolicy',  # RAM observation works best with MLP policy
    'learning_rate': 1e-4,
    'n_steps': 256,
    'batch_size': 64,
    'n_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'total_timesteps': 200_000,
    'save_freq': 1_000,
    'save_path': 'models/best/ppo_mario_ram',
    'tensorboard_log': 'runs/',
    'device': 'auto',
}

LOW_SPEC_NOTES = {
    'gpu': 'GTX 1650 4GB',
    'cpu': 'Old low-frequency desktop CPU',
    'strategy': [
        'Use gym-retro baseline first.',
        'Single-thread training.',
        '84x84 grayscale + frame stack.',
        'Render only in play mode.',
    ],
}
