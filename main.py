import inputs

from MyEnv.MyEnv import MyEnv
import time
from inputs import get_gamepad
import threading
import math
import numpy as np


class ManualControl:
    def __init__(self):
        self.x = 0
        self.y = 0

        t1 = threading.Thread(target=self.control_thread)
        t1.daemon = True
        t1.start()

        self.main()

    def main(self):
        env = MyEnv(render=True, step_time=0.02)
        while True:
            env.step([self.x, self.y])
            #print(env.drone)
            #print(self.calculate_distance(np.array([-1, -1]), 1, 0.173533))

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
