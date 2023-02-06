from MyEnv.MyEnv import MyEnv
from stable_baselines3 import PPO
import torch as th
from stable_baselines3.common.callbacks import CheckpointCallback


env = MyEnv(render=False, step_time=0.02)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[256, 256])])

model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="tensorboard")

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='models', name_prefix='9')

model.learn(total_timesteps=2_300_000, callback=checkpoint_callback, log_interval=5)

model.save("models/ppo9")
