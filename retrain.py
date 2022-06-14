from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO
import torch as th

env = MyEnv(render=False, step_time=0.02, laser_noise=(0, 0.01))

model = PPO.load("ppo15")

model.set_env(env)

model.learn(total_timesteps=2_000_000)

model.save("ppo16")