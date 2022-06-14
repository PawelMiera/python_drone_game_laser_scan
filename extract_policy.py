from stable_baselines3 import PPO

filename = "m_360_61"

model = PPO.load(filename)

model.policy.save(filename + "_policy.zip")
