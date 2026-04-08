# SuperMario
A improved supermario game based on https://github.com/justinmeister/Mario-Level-1.
* support four levels：level 1-1 to level 1-4 
* support go into the pipe
* use json file to store level data (e.g. position of enemy, brick, box and pipe)
* add new enemies in level 1-3 and 1-4 
* add slider in level 1-2

# Requirement
* Python 3.7
* Python-Pygame 1.9

# How To Start Game
$ python main.py

# AI Training (PPO, multi-Mario parallel)
## Install AI dependencies
```bash
python -m pip install -r requirements-ai.txt
```

## Train with multiple parallel environments
```bash
python -m ai.train_ppo --n-envs 8 --total-timesteps 1000000 --level 1
```

## Watch trained model play
```bash
python -m ai.play_ppo --model checkpoints/ppo_mario_final.zip --episodes 5
```

Notes:
- `--n-envs` controls how many Marios train in parallel (PPO VecEnv).
- Increase `--n-envs` and `--total-timesteps` for better performance.
- Use TensorBoard:
```bash
tensorboard --logdir logs
```

# How to Play
* use LEFT/RIGHT/DOWN key to control player
* use key 'a' to jump
* use key 's' to shoot firewall or run

# Demo
![level_1_1](https://raw.githubusercontent.com/marblexu/PythonSuperMario/master/resources/demo/level_1_1.png)
![level_1_2](https://raw.githubusercontent.com/marblexu/PythonSuperMario/master/resources/demo/level_1_2.png)
![level_1_3](https://raw.githubusercontent.com/marblexu/PythonSuperMario/master/resources/demo/level_1_3.png)
![level_1_4](https://raw.githubusercontent.com/marblexu/PythonSuperMario/master/resources/demo/level_1_4.png)
