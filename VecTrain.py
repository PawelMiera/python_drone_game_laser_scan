from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import torch as th
from MyEnv.MyEnv import MyEnv

def make_env(env_id, rank, seed=0):
    def _init():
        env = MyEnv(render=False, step_time=0.02, laser_noise=(0, 0.01))

        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 40  # Number of processes to use
    # Create the vectorized environment
    envs = [make_env(env_id, i) for i in range(num_cpu)]
    env = SubprocVecEnv(envs)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[128, 128], vf=[256, 256])])

    model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="tensorboard")
    model.learn(total_timesteps=5_000_000)

    model.save("m_360_81")
