from math import sqrt, pi, atan2, sin, cos, copysign, tan

from DroneState.DroneState import DroneState
from Tree.Tree import Tree
import numpy as np
import cv2
import time
import random
import gym
from gym import spaces

class MyEnv(gym.Env):
    def __init__(self, render, step_time):

        super(MyEnv, self).__init__()

        """""""""""sim"""""""""""
        self.last_time = time.time()
        self.do_render = render
        self.step_time = step_time
        self.drone = DroneState(max_speed=3, acc_x=1.5, acc_y=10.5, step_time=step_time)
        self.window_size = (1000, 1000)
        self.window_size_half = (int(self.window_size[0] / 2), int(self.window_size[1] / 2))
        self.pixels_per_meter = 100



        """""""""""grids"""""""""""
        self.grid_size = 10
        self.grid_size_half = self.grid_size / 2
        self.current_grid = [0, 0]
        self.last_grid = [0, 0]

        self.tree_radius_range = (0.2, 0.65)
        self.trees_per_grid = 20
        self.trees_min_distance = 0.8

        self.trees = {}
        self.closest_trees = []

        """""""""""laser"""""""""""

        self.laser_max_range = 5.0
        self.laser_resolution = 315
        self.laser_angle_per_step = 2 * pi / self.laser_resolution

        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range)

        self.generate_new_trees()

        self.action_space = spaces.Box(low=-1, high=1, shape=([2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=([self.laser_resolution]), dtype=np.float32)

    def step(self, actions):
        now = time.time()
        # print("Act ", actions)
        # print(now - self.last_time)
        self.last_time = now
        self.drone.make_step(actions)
        self.current_grid = [int(self.drone.pos[0] / self.grid_size), int(self.drone.pos[1] / self.grid_size)]

        if self.current_grid != self.last_grid:
            self.generate_new_trees()

        self.get_closest_trees()

        self.calculate_laser_distances()

        obs = self.get_obs()

        reward = self.computeReward()

        done = self.isDone()

        if self.isDone():
            reward = 1

        if self.do_render:
            self.render()

        self.last_grid = self.current_grid

        return obs, reward, done, {}

    def get_obs(self):
        return ((self.laser_ranges.copy() - 2.5) * 0.4).astype(np.float32)

    def computeReward(self):
        reward = 0
        dist_margin = 0.1

        for tree in self.closest_trees:
            dist = tree.distance(self.drone.pos)
            if dist - tree.r - dist_margin < 0:
                reward += -0.2

        reward += 0.01 * self.drone.speed_x

        return reward

    def reset(self):
        self.trees.clear()
        self.drone.pos = np.array([0, 0]).astype(np.float32)
        self.current_grid = [0, 0]
        self.last_grid = [0, 0]
        self.generate_new_trees()

        return self.get_obs()

    def isDone(self):
        if self.drone.pos[0] > 100:
            return True
        return False

    def close(self):
        raise NotImplementedError

    def calculate_laser_distances(self):
        for a in range(self.laser_resolution):
            angle = a * self.laser_angle_per_step
            dist = 5.0
            for tree in self.closest_trees:
                dist_to_drone = tree.distance(self.drone.pos)
                if dist_to_drone - tree.r < self.laser_max_range and dist_to_drone > tree.r:
                    current_dist = self.calculate_one_laser_distance(-(tree.p - self.drone.pos), tree.r,
                                                                     angle)

                    dist = min(dist, current_dist)

            self.laser_ranges[a] = dist

    def angle_in_range(self, angle, first, second):
        return (
                self.sign(self.cross_product(first, angle)) ==
                self.sign(self.cross_product(angle, second)) ==
                self.sign(self.cross_product(first, second))
        )

    def cross_product(self, first, second):
        first_x = cos(first)
        first_y = sin(first)

        second_x = cos(second)
        second_y = sin(second)

        return first_x * second_y - first_y * second_x

    def sign(self, x):
        return copysign(1, x)

    def calculate_one_laser_distance(self, pos, r, angle):

        angle_direction = atan2(-pos[1], -pos[0])

        angle_direction_max = angle_direction - pi / 4
        angle_direction_min = angle_direction + pi / 4

        dist = self.laser_max_range

        if self.angle_in_range(angle, angle_direction_min, angle_direction_max):
            a_l = tan(angle)
            b_l = pos[1] - pos[0] * a_l
            a2_l = a_l * a_l
            r2 = r * r
            b2_l = b_l * b_l

            a_q = a2_l + 1
            b_q = 2 * a_l * b_l
            c_q = b2_l - r2

            d = b_q * b_q - 4 * a_q * c_q

            if d > 0:
                x1 = (-b_q - sqrt(d)) / (2 * a_q)
                x2 = (-b_q + sqrt(d)) / (2 * a_q)

                y1 = a_l * x1 + b_l
                y2 = a_l * x2 + b_l

                dist = self.calculate_distance(np.array([x1, y1]), pos)

                dist = min(dist, self.calculate_distance(np.array([x2, y2]), pos))

            elif d == 0:
                x1 = - b_q / (2 * a_q)
                y1 = a_l * x1 + b_l
                self.calculate_distance(np.array([x1, y1]), pos)

        return dist

    def calculate_distance(self, p1, p2):
        return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))

    def get_closest_trees(self):
        self.closest_trees.clear()

        for i in range(self.current_grid[0] - 1, self.current_grid[0] + 1 + 1):
            for j in range(self.current_grid[1] - 1, self.current_grid[1] + 1 + 1):
                key = str([i, j])
                if key in self.trees:
                    current_closest_trees = [tree for tree in self.trees[key] if
                                             tree.distance(self.drone.pos) - tree.r <= self.grid_size / 2]

                    self.closest_trees += current_closest_trees

    def generate_new_trees(self):
        for i in range(self.current_grid[0] - 1, self.current_grid[0] + 1 + 1):
            for j in range(self.current_grid[1] - 1, self.current_grid[1] + 1 + 1):
                key = str([i, j])
                if key not in self.trees:
                    accepted_trees = []

                    while len(accepted_trees) < self.trees_per_grid:
                        pos = np.array([random.uniform(-self.grid_size_half, self.grid_size_half),
                                        random.uniform(-self.grid_size_half, self.grid_size_half)])
                        pos += np.array([i, j]) * self.grid_size
                        radius = random.uniform(self.tree_radius_range[0], self.tree_radius_range[1])
                        current_tree = Tree(pos, radius)
                        too_close = False

                        if current_tree.distance(np.array([0, 0])) < 2.5:
                            continue

                        for accepted_tree in accepted_trees:
                            if current_tree.distance(accepted_tree.p) - current_tree.r - accepted_tree.r < \
                                    self.trees_min_distance:
                                too_close = True
                                break

                        if too_close:
                            continue

                        for k in range(self.current_grid[0] - 1, self.current_grid[0] + 1 + 1):
                            for l in range(self.current_grid[1] - 1, self.current_grid[1] + 1 + 1):
                                key1 = str([k, l])
                                if key1 in self.trees:
                                    existing_trees = self.trees[key1]

                                    for existing_tree in existing_trees:
                                        if current_tree.distance(existing_tree.p) - current_tree.r - existing_tree.r < \
                                                self.trees_min_distance:
                                            too_close = True
                                            break
                        if too_close:
                            continue

                        accepted_trees.append(current_tree)
                    self.trees[key] = accepted_trees

    def render(self, mode='human'):
        start_time = time.time()
        background = np.zeros((self.window_size[0], self.window_size[1], 3), np.uint8)
        background[:] = (0, 255, 0)

        for tree in self.closest_trees:
            pos_diff = (tree.p - self.drone.pos) * self.pixels_per_meter

            pos_diff = np.array(
                [self.window_size_half[1] + pos_diff[1], self.window_size_half[0] - pos_diff[0]]).astype(np.int)

            radius = int(tree.r * self.pixels_per_meter)

            cv2.circle(background, pos_diff, radius, (60, 103, 155), -1)

        for i in range(self.laser_resolution):
            A = int(self.laser_ranges[i] * self.pixels_per_meter * sin(i * self.laser_angle_per_step)) + \
                self.window_size_half[0]
            B = -int(self.laser_ranges[i] * self.pixels_per_meter * cos(i * self.laser_angle_per_step)) + \
                self.window_size_half[1]
            cv2.line(background, self.window_size_half, (A, B), (0, 0, 255), 1)
            cv2.circle(background, (A, B), 2, (0, 0, 255), -1)

        drone_x_min = self.window_size_half[0] - 15
        drone_y_min = self.window_size_half[1] - 10
        background = cv2.rectangle(background, (drone_y_min, drone_x_min), (drone_y_min + 20, drone_x_min + 30),
                                   (0, 0, 0), -1)
        background = cv2.rectangle(background, (drone_y_min, drone_x_min), (drone_y_min + 20, drone_x_min + 5),
                                   (255, 0, 0), -1)

        cv2.imshow("game", background)
        cv2.waitKey(1)

        # wait_time = self.step_time - (time.time() - start_time)
        # print("wait ", wait_time)
        # if wait_time > 0:
        #     time.sleep(wait_time)
