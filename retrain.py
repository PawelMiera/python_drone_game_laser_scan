from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='models',
                                         name_prefix='best_retrain2')

env = MyEnv(render=False, step_time=0.02, laser_noise=None, laser_disturbtion=False)

model = PPO.load("models/new_best.zip")

model.set_env(env)

model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

model.save("ppo16")