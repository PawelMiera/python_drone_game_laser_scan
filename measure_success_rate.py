from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO

env = MyEnv(render=False, step_time=0.02, laser_noise=(0, 0.01), laser_disturbtion=True)

model = PPO.load("m_360_61.zip")

obs = env.reset()

suc = 0
col = 0
out = 0

number_of_runs = 100
current_run = 0

while current_run < number_of_runs:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)

    if dones:
        print("Reward", rewards, "Run", current_run)

        if rewards == 10:
            suc += 1
        elif rewards == -1:
            col += 1
        elif rewards == -10:
            out += 1

        obs = env.reset()

        current_run += 1

    # env.render()

print(suc, col, out)

print("Success_rate: ", suc / (col + out))
