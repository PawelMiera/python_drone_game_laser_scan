from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO
import torch as th

env = MyEnv(render=False, step_time=0.02, laser_noise=(0, 0.01))

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[256, 256])])

model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="tensorboard")
model.learn(total_timesteps=2_000_000)

model.save("ppo13")
