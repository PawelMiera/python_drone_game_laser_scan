from stable_baselines3 import PPO

filename = "vec_model6"

model = PPO.load(filename)

model.policy.save(filename + "_policy")
