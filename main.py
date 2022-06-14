import inputs

from MyEnv.MyEnv import MyEnv
import time
from inputs import get_gamepad
import threading
import math
import numpy as np
# from stable_baselines3.common.env_checker import check_env
#
# from stable_baselines3 import PPO
import pandas

class ManualControl:
    def __init__(self):
        self.x = 0
        self.y = 0

        # t1 = threading.Thread(target=self.control_thread)
        # t1.daemon = True
        # t1.start()

        self.main()

    def main(self):
        env = MyEnv(render=True, step_time=0.02, laser_noise=(0, 0.05))
        start = time.time()
        i = 0
        data = []
        while i < 1000:
            if i < 300:
                env.step([0, 1])
            else:
                env.step([0, -1])
            #print(env.drone)
            #print(time.time() - start)
            #print(self.calculate_distance(np.array([-1, -1]), 1, 0.173533))

            time_control = np.array([float(i * 0.02), env.drone.pos[0],
                                     env.drone.pos[1],
                                     env.drone.speed[0],
                                     env.drone.speed[1],
                                     env.drone.acc_x,
                                     env.drone.acc_y])
            data.append(time_control)
            i += 1
        df = pandas.DataFrame(np.array(data),
                              columns=["time", "p_x", "p_y", "v_x", "v_y", "a_x", "a_y"])
        df.to_csv("my.csv")

    def control_thread(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_X":
                    self.y = event.state / 32768
                elif event.code == "ABS_Y":
                    self.x = event.state / 32768



if __name__ == '__main__':
    con = ManualControl()
    # env = MyEnv(render=False, step_time=0.02)
    # #
    # model = PPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=100_000)
    #
    #
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()