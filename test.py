from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO


env = MyEnv(render=False, step_time=0.02, laser_noise=None, laser_disturbtion=False, collision_is_crash=True)

model = PPO.load("models/best_retrain2_400000_steps.zip")

obs = env.reset()
for _ in range(8000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)

    env.render()

    if dones:
        print("reward", rewards)
        obs = env.reset()

