from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO
import torch as th

env = MyEnv(render=False, step_time=0.02)

model = PPO.load("ppo3")

model.set_env(env)

model.learn(total_timesteps=500_000)

model.save("ppo4")