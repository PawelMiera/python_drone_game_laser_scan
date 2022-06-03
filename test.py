from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO
import torch as th

print("OK")
env = MyEnv(render=False, step_time=0.02, test=True)

print("OK1")
model = PPO.load("ppo3")

obs = env.reset()
for _ in range(4000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
