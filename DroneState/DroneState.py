import numpy as np

class DroneState:
    def __init__(self, max_speed, acc_x, acc_y, step_time):
        self.max_acc_x = acc_x
        self.max_acc_y = acc_y
        self.min_acc_x = -acc_x
        self.min_acc_y = -acc_y
        self.max_speed = max_speed
        self.pos = np.array([0, 0]).astype(np.float32)
        self.speed_x = 0
        self.speed_y = 0
        self.req_speed_x = 0
        self.req_speed_y = 0
        self.step_time = step_time

        self.time = 0

    def make_step(self, req_speed):
        self.req_speed_x = req_speed[0] * self.max_speed
        self.req_speed_y = req_speed[1] * self.max_speed

        speed_diff_x = self.req_speed_x - self.speed_x
        speed_diff_y = self.req_speed_y - self.speed_y

        current_acc_x = min(max(speed_diff_x / self.step_time, self.min_acc_x), self.max_acc_x)
        current_acc_y = min(max(speed_diff_y / self.step_time, self.min_acc_y), self.max_acc_y)

        self.speed_x += current_acc_x * self.step_time
        self.speed_y += current_acc_y * self.step_time

        new_pos_x = self.speed_x * self.step_time + current_acc_x * self.step_time * self.step_time / 2
        new_pos_y = self.speed_y * self.step_time + current_acc_y * self.step_time * self.step_time / 2

        self.pos += np.array([new_pos_x, new_pos_y]).astype(np.float32)
        self.time += self.step_time

    def __str__(self):
        return "Time: {:.2f}: P: [{:.2f}, {:.2f}], V: [{:.2f}, {:.2f}], R_V: [{:.2f}, {:.2f}]".format(self.time, self.pos[0],
                                                                                                      self.pos[1],
                                                                                                      self.speed_x,
                                                                                                      self.speed_y,
                                                                                                      self.req_speed_x,
                                                                                                      self.req_speed_y)
