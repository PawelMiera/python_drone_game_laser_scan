from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO


env = MyEnv(render=False, step_time=0.02, laser_noise=(0, 0.01), laser_disturbtion=False)

model = PPO.load("m_360_61.zip")

obs = env.reset()
for _ in range(8000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)

    env.render()

    if dones:
        obs = env.reset()

