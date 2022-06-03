import numpy as np
# from numba import int32, float32    # import the types
#
#
# spec = [
#     ('max_acc_x', float32),
#     ('max_acc_y', float32),
#     ('min_acc_x', float32),
#     ('min_acc_y', float32),
#     ('acc_gain', float32),
#     ('max_speed', float32),
#     ('pos', float32[:]),
#     ('speed_x', float32),
#     ('speed_y', float32),
#     ('acc_x', float32),
#     ('acc_y', float32),
#     ('acc_gain', float32),
#     ('req_speed_x', float32),
#     ('req_speed_y', float32),
#     ('req_speed_y', float32),
#     ('step_time', float32),
#     ('time', float32),
# ]


class DroneState:
    def __init__(self, max_speed, max_acc_x, max_acc_y, step_time):
        self.max_acc_x = max_acc_x
        self.max_acc_y = max_acc_y
        self.min_acc_x = -max_acc_x
        self.min_acc_y = -max_acc_y
        self.max_speed = max_speed
        self.pos = np.array([0, 0]).astype(np.float32)
        self.speed = np.array([0, 0]).astype(np.float32)
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.acc_gain = 1.3
        self.req_speed = np.array([0, 0]).astype(np.float32)
        self.step_time = step_time

        self.time = 0

    def reset(self):
        self.pos = np.array([0, 0]).astype(np.float32)
        self.speed = np.array([0, 0]).astype(np.float32)
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.time = 0
        self.req_speed = np.array([0, 0]).astype(np.float32)

    def make_step(self, req_speed):
        self.req_speed = np.multiply(req_speed, self.max_speed)

        speed_diff = self.req_speed - self.speed

        self.acc_x = min(max(speed_diff[0] / self.step_time * self.acc_gain, self.min_acc_x), self.max_acc_x)
        self.acc_y = min(max(speed_diff[1] / self.step_time * self.acc_gain, self.min_acc_y), self.max_acc_y)

        self.speed[0] += self.acc_x * self.step_time
        self.speed[1] += self.acc_y * self.step_time

        self.pos[0] = self.pos[0] + self.speed[0] * self.step_time + self.acc_x * self.step_time * self.step_time / 2
        self.pos[1] = self.pos[1] + self.speed[1] * self.step_time + self.acc_y * self.step_time * self.step_time / 2

        self.time += self.step_time


    def __str__(self):
        return "Time: {:.2f}: P: [{:.2f}, {:.2f}], V: [{:.2f}, {:.2f}], R_V: [{:.2f}, {:.2f}], A: [{:.2f}, {:.2f}]".format(
            self.time, self.pos[0],
            self.pos[1],
            self.speed[0],
            self.speed[1],
            self.req_speed[0],
            self.req_speed[1], self.acc_x, self.acc_y)
