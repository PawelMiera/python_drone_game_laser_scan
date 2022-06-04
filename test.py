from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO
import torch as th


env = MyEnv(render=False, step_time=0.02, test=True, laser_noise=(0, 0.05))

model = PPO.load("ppo5")

obs = env.reset()
for _ in range(4000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
