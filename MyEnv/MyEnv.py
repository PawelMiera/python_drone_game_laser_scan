from math import sqrt, pi, sin, cos

from DroneState.DroneState import DroneState
import numpy as np
import cv2
import random
import gym
from gym import spaces


class MyEnv(gym.Env):
    def __init__(self, render, step_time, test=False, max_speed=1, max_acc_x=0.6, max_acc_y=0.6, laser_resolution=360,
                 laser_range_max=10, laser_range_min=0.15, laser_noise=(0, 0.01), laser_disturbtion=True,
                 laser_update_rate=10):

        super(MyEnv, self).__init__()

        """""""""""sim"""""""""""
        self.do_render = render
        self.step_time = step_time
        self.test = test

        self.drone = DroneState(max_speed=max_speed, max_acc_x=max_acc_x, max_acc_y=max_acc_y, step_time=step_time)

        self.window_size = (1000, 1000)
        self.window_size_half = (int(self.window_size[0] / 2), int(self.window_size[1] / 2))
        self.pixels_per_meter = 50

        """""""""""grids"""""""""""
        self.grid_size = 16
        self.grid_size_half = self.grid_size / 2
        self.current_grid = [0, 0]
        self.last_grid = [0, 0]

        self.tree_radius_range = (0.15, 0.55)
        self.trees_per_grid_range = (30, 60)
        self.trees_per_grid = int(random.uniform(self.trees_per_grid_range[0], self.trees_per_grid_range[1]))
        self.trees_min_distance = 0.8

        self.world_size_in_grids = np.array([-2, 3, -2, 3])
        self.world_size = np.multiply(self.world_size_in_grids, self.grid_size)
        self.trees_array = np.zeros(
            (self.world_size_in_grids[1] - self.world_size_in_grids[0] + 2,
             self.world_size_in_grids[3] - self.world_size_in_grids[2] + 2, self.trees_per_grid, 3),
            dtype=np.float32)

        self.mid_grid = [-self.world_size_in_grids[0] + 1, -self.world_size_in_grids[2] + 1]
        self.current_grid = self.mid_grid.copy()
        self.last_grid = self.mid_grid.copy()

        self.closest_trees = np.array([]).reshape(0, 3)
        self.closest_distance = np.array([]).reshape(0, 1)
        self.near_grid_trees = np.array([]).reshape(0, 3).astype(np.float32)

        """""""""""laser"""""""""""

        self.laser_max_range = laser_range_max
        self.laser_min_range = laser_range_min
        self.laser_resolution = laser_resolution
        self.laser_angle_per_step = 2 * pi / self.laser_resolution
        self.laser_noise = laser_noise
        self.laser_disturbtion = laser_disturbtion
        self.laser_last_update_time = 0

        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)

        angles = []
        self.pi_2_positions = []
        for i in range(self.laser_resolution):
            val = round(i * self.laser_angle_per_step, 10)
            angles.append(val)
            pi1 = round(pi / 2, 10)
            pi2 = round(3 * pi / 2, 10)
            if val == pi1 or val == pi2:
                self.pi_2_positions.append(i)

        self.laser_angles = np.array(angles).astype(np.float32)
        self.laser_angles_sin = np.sin(self.laser_angles)
        self.laser_angles_cos = np.cos(self.laser_angles)
        self.laser_tangents = np.tan(self.laser_angles)
        self.laser_tangents_2 = np.multiply(self.laser_tangents, 2)

        self.laser_update_time = 1 / laser_update_rate

        self.a_q = np.add(np.square(self.laser_tangents), 1)
        self.a_q_2 = np.multiply(self.a_q, 2)

        self.generate_new_trees()
        self.update_near_trees()

        self.action_space = spaces.Box(low=-1, high=1, shape=([2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=([self.laser_resolution]), dtype=np.float32)

        self.current_step = 0
        self.crash = False

    def step(self, actions):
        current_time = self.step_time * self.current_step

        self.drone.make_step(actions)
        self.current_grid = [int(round(self.drone.pos[0] / self.grid_size + self.mid_grid[0])),
                             int(round(self.drone.pos[1] / self.grid_size + self.mid_grid[1]))]

        if self.current_grid != self.last_grid:
            self.update_near_trees()

        self.get_closest_trees()

        if current_time - self.laser_last_update_time > self.laser_update_time:
            self.laser_last_update_time = current_time
            self.calculate_laser_distances()

        obs = self.get_obs()

        reward = self.computeReward()

        done, c_reward = self.isDone()

        if self.do_render:
            self.render()

        self.last_grid = self.current_grid

        if self.test and self.crash:
            print("Crash")

        if done:
            reward = c_reward
            if self.test:
                print("Reward", reward)
                self.reset()

        self.current_step += 1

        return obs, reward, done, {}

    def get_obs(self):
        divider = self.laser_max_range / 2
        return ((self.laser_ranges.copy() - divider) / divider).astype(np.float32)

    def computeReward(self):
        dist_margin = 0.25
        colision_reward = 0
        min_dist = self.laser_max_range
        for tree in self.closest_trees:
            dist = self.calculate_distance(self.drone.pos, tree[:2])
            if dist - tree[2] - dist_margin < 0:
                colision_reward += -0.25
            if dist - tree[2] - self.laser_min_range < 0:
                colision_reward = -1
                min_dist = 0
                break

            if dist - tree[2] < min_dist:
                min_dist = dist
        dist_reward = 0.0 * min_dist

        speed_reward = 0.8 * self.drone.speed[0]

        pos_y_offset_penalty = -0.12 * abs(self.drone.pos[1])

        if speed_reward < 0:
            speed_reward *= 3

        reward = colision_reward + speed_reward + pos_y_offset_penalty + dist_reward

        return reward

    def reset(self):
        self.drone.reset()
        self.current_grid = self.mid_grid.copy()
        self.last_grid = self.mid_grid.copy()

        self.trees_per_grid = int(random.uniform(self.trees_per_grid_range[0], self.trees_per_grid_range[1]))

        self.trees_array = np.zeros(
            (self.world_size_in_grids[1] - self.world_size_in_grids[0] + 2,
             self.world_size_in_grids[3] - self.world_size_in_grids[2] + 2, self.trees_per_grid, 3),
            dtype=np.float32)

        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)

        self.generate_new_trees()

        self.update_near_trees()

        self.get_closest_trees()

        self.calculate_laser_distances()
        self.laser_last_update_time = 0
        self.current_step = 0
        self.crash = False

        return self.get_obs()

    def isDone(self):
        if self.drone.pos[0] > self.world_size[1]:
            return True, 10

        if self.drone.pos[0] < self.world_size[0] or self.drone.pos[1] < self.world_size[2] or self.drone.pos[1] > \
                self.world_size[3]:
            return True, -10

        if self.step_time * self.current_step > 80:
            return True, -10

        return False, 0

    def close(self):
        raise NotImplementedError

    def calculate_distance(self, p1, p2):
        return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))

    def render(self, mode='human'):
        background = np.zeros((self.window_size[0], self.window_size[1], 3), np.uint8)
        background[:] = (0, 255, 0)

        for tree in self.closest_trees:
            pos_diff = (tree[:2] - self.drone.pos) * self.pixels_per_meter

            pos_diff = np.array(
                [self.window_size_half[1] + pos_diff[1], self.window_size_half[0] - pos_diff[0]]).astype(np.int)

            radius = int(tree[2] * self.pixels_per_meter)

            cv2.circle(background, tuple(pos_diff), radius, (60, 103, 155), -1)

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

    def are_angles_not_between(self, angles_sin, angles_cos, max_sin, max_cos, min_sin, min_cos):

        s1 = np.sign(np.multiply(min_cos, angles_sin) - np.multiply(min_sin, angles_cos))
        s2 = np.sign(np.multiply(angles_cos, max_sin) - np.multiply(angles_sin, max_cos))
        s3 = np.stack([np.sign(np.multiply(min_cos, max_sin) - np.multiply(min_sin, max_cos))] * self.laser_resolution)

        return np.logical_not(np.logical_and((s1 == s2), (s2 == s3)))

    def calculate_laser_distances(self):
        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range, dtype=np.float32)

        trees_relative_pos = self.closest_trees[:, :2] - self.drone.pos
        trees_minus_relative_pos = -trees_relative_pos
        tree_angle_direction = np.arctan2(trees_relative_pos[:, 1], trees_relative_pos[:, 0])

        angle_direction_max = np.subtract(tree_angle_direction, pi / 4)
        angle_direction_min = np.add(tree_angle_direction, pi / 4)

        sin_max = np.sin(angle_direction_max)
        cos_max = np.cos(angle_direction_max)
        sin_min = np.sin(angle_direction_min)
        cos_min = np.cos(angle_direction_min)

        if len(self.closest_trees) > 0:
            all_laser_sin = np.stack([self.laser_angles_sin] * len(tree_angle_direction), axis=1)
            all_laser_cos = np.stack([self.laser_angles_cos] * len(tree_angle_direction), axis=1)

            angles_result = self.are_angles_not_between(all_laser_sin, all_laser_cos, sin_max, cos_max, sin_min,
                                                        cos_min)

            trees_r2 = np.square(self.closest_trees[:, 2])
        self.crash = False
        for i in range(len(self.closest_trees)):
            if self.closest_distance[i] > self.closest_trees[i, 2]:
                available_laser_tangents = self.laser_tangents.copy()
                available_laser_tangents[angles_result[:, i]] = np.nan
                b_l = trees_minus_relative_pos[i, 1] - (trees_minus_relative_pos[i, 0] * available_laser_tangents)
                b2_l = np.square(b_l)

                b_q = np.multiply(self.laser_tangents_2, b_l)
                c_q = b2_l - trees_r2[i]

                d = np.square(b_q) - np.multiply(self.a_q * c_q, 4)

                sqrt_d = np.sqrt(d)

                x1 = (- b_q - sqrt_d) / self.a_q_2
                x2 = (- b_q + sqrt_d) / self.a_q_2

                y1 = self.laser_tangents * x1 + b_l
                y2 = self.laser_tangents * x2 + b_l

                p1 = np.stack((x1, y1), axis=1)
                p2 = np.stack((x2, y2), axis=1)

                dist_array_1 = np.sqrt(np.sum(np.square(p1 - trees_minus_relative_pos[i, :2]), axis=1))
                dist_array_2 = np.sqrt(np.sum(np.square(p2 - trees_minus_relative_pos[i, :2]), axis=1))
                for p in self.pi_2_positions:
                    if not np.isnan(available_laser_tangents[p]):
                        try:
                            y = sqrt(trees_r2[i] - trees_minus_relative_pos[i, 0] * trees_minus_relative_pos[i, 0])
                            d = min(abs(trees_minus_relative_pos[i, 1] - y), abs(trees_minus_relative_pos[i, 1] + y))
                            dist_array_1[p] = d
                            dist_array_2[p] = d
                        except ValueError:
                            dist_array_1[p] = self.laser_max_range
                            dist_array_2[p] = self.laser_max_range

                min_dist = np.nan_to_num(np.minimum(dist_array_1, dist_array_2), nan=self.laser_max_range)
                self.laser_ranges = np.minimum(min_dist, self.laser_ranges)
                self.crash = False
            else:
                self.laser_ranges = np.full(self.laser_resolution, self.laser_min_range, dtype=np.float32)
                self.crash = True
                break

        inf_mask = self.laser_ranges == self.laser_max_range
        not_inf_mask = np.logical_not(inf_mask)
        if self.laser_noise is not None and not self.crash:
            noise = np.random.normal(self.laser_noise[0], self.laser_noise[1], size=np.sum(not_inf_mask)).astype(
                np.float32)
            self.laser_ranges[not_inf_mask] += noise
            self.laser_ranges = np.maximum(self.laser_ranges, self.laser_min_range)

        self.laser_ranges[inf_mask] = 0

        if self.laser_disturbtion and not self.crash:
            disturbtion_mask = np.random.choice(np.arange(self.laser_ranges.size), replace=False,
                                       size=25)
            self.laser_ranges[disturbtion_mask] = 0


    def update_near_trees(self):
        self.near_grid_trees = np.array([]).reshape(0, 3).astype(np.float32)
        for m in range(self.current_grid[0] - 1, self.current_grid[0] + 2):
            for n in range(self.current_grid[1] - 1, self.current_grid[1] + 2):
                if 0 <= m < self.trees_array.shape[0] and 0 <= n < self.trees_array.shape[1]:
                    self.near_grid_trees = np.concatenate((self.near_grid_trees, self.trees_array[m, n]))

    def get_closest_trees(self):
        self.closest_trees = np.array([]).reshape(0, 3).astype(np.float32)
        self.closest_distance = np.array([]).reshape(0, 3).astype(np.float32)

        if len(self.near_grid_trees > 0):
            near_grid_trees_points = self.near_grid_trees[:, :2].copy()
            near_grid_trees_radius = self.near_grid_trees[:, 2].copy()
            dist_array = np.sqrt(np.sum(np.square(near_grid_trees_points - self.drone.pos), axis=1))
            close_array = np.subtract(dist_array - near_grid_trees_radius, self.laser_max_range)

            near_trees = self.near_grid_trees[close_array < 0]
            self.closest_distance = dist_array[close_array < 0]
            self.closest_trees = near_trees

    def generate_new_trees(self):
        for i in range(self.trees_array.shape[0]):
            for j in range(self.trees_array.shape[1]):
                accepted_trees = []
                near_grid_trees = np.array([]).reshape(0, 3)
                for m in range(i - 1, i + 2):
                    for n in range(j - 1, j + 2):
                        if not (m == i and n == j):
                            if 0 <= m < self.trees_array.shape[0] and 0 <= n < self.trees_array.shape[1]:
                                near_grid_trees = np.concatenate((near_grid_trees, self.trees_array[m, n]))

                near_grid_trees = near_grid_trees[~np.all(near_grid_trees == 0, axis=1)]
                tries = 0
                while len(accepted_trees) < self.trees_per_grid:
                    if tries > 1000:
                        print("Cant create all trees")
                        while len(accepted_trees) < self.trees_per_grid:
                            accepted_trees.append(accepted_trees[0])
                        break
                    tries += 1
                    p1 = random.uniform(-self.grid_size_half, self.grid_size_half) + (
                            i - self.mid_grid[0]) * self.grid_size
                    p2 = random.uniform(-self.grid_size_half, self.grid_size_half) + (
                            j - self.mid_grid[1]) * self.grid_size
                    radius = random.uniform(self.tree_radius_range[0], self.tree_radius_range[1])

                    current_tree = np.array([p1, p2, radius])

                    if self.calculate_distance(current_tree[:2], np.array([0, 0])) < 2.5:
                        continue

                    accepted_array = np.array(accepted_trees)
                    if len(accepted_array > 0):
                        accepted_points = accepted_array[:, :2].copy()
                        accepted_radius = accepted_array[:, 2].copy()
                        dist_array = np.sqrt(np.sum(np.square(accepted_points - current_tree[:2]), axis=1))
                        close_array = np.subtract(dist_array - accepted_radius, radius + self.trees_min_distance)

                        if np.any(close_array < 0):
                            continue

                    if len(near_grid_trees > 0):
                        near_grid_trees_points = near_grid_trees[:, :2].copy()
                        near_grid_trees_radius = near_grid_trees[:, 2].copy()
                        dist_array = np.sqrt(np.sum(np.square(near_grid_trees_points - current_tree[:2]), axis=1))
                        close_array = np.subtract(dist_array - near_grid_trees_radius, radius + self.trees_min_distance)

                        if np.any(close_array < 0):
                            continue
                    tries = 0
                    accepted_trees.append(current_tree)

                self.trees_array[i, j] = np.array(accepted_trees)
