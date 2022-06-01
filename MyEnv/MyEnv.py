import math

from DroneState.DroneState import DroneState
from Tree.Tree import Tree
import numpy as np
import cv2
import time
import random


class MyEnv():
    def __init__(self, render, step_time):

        """""""""""sim"""""""""""

        self.do_render = render
        self.step_time = step_time
        self.drone = DroneState(max_speed=3, acc_x=1.5, acc_y=10.5, step_time=step_time)
        self.window_size = (1000, 1000)
        self.mid_point = (int(self.window_size[0] / 2), int(self.window_size[1] / 2))
        self.pixels_per_meter = 100


        """""""""""grids"""""""""""
        self.grid_size = 10
        self.grid_size_half = self.grid_size / 2
        self.current_grid = [0, 0]
        self.last_grid = [0, 0]
        self.generated_grids = []

        self.tree_radius_range = (0.2, 0.65)
        self.trees_per_grid = 20
        self.trees_min_distance = 0.8

        self.trees = {}
        self.trees[str([0, 0])] = [Tree(np.array([3,3]), 1)]

        """""""""""laser"""""""""""

        self.laser_max_range = 5.0
        self.laser_resolution = 360
        self.laser_angle_per_step = 2 * math.pi / self.laser_resolution

        self.laser_ranges = np.full(self.laser_resolution, self.laser_max_range)

        self.generate_new_trees()

    def reset(self):
        pass

    def calculate_laser_distances(self):
        for a in range(self.laser_resolution):
            angle = a * self.laser_angle_per_step
            dist = 5.0
            for i in range(self.current_grid[0] - 1, self.current_grid[0] + 1 + 1):
                for j in range(self.current_grid[1] - 1, self.current_grid[1] + 1 + 1):
                    key = str([i, j])
                    if key in self.trees:
                        for tree in self.trees[key]:
                            if tree.distance(self.drone.pos) - tree.r < self.laser_max_range:
                                current_dist = self.calculate_one_laser_distance(-(tree.p - self.drone.pos), tree.r, angle)

                                dist = min(dist, current_dist)

            self.laser_ranges[a] = dist



    def calculate_one_laser_distance(self, pos, r, angle):
        a_l = math.tan(angle)
        b_l = pos[0] - pos[1] * a_l

        a2_l = a_l * a_l
        r2 = r * r
        b2_l = b_l * b_l

        a_q = a2_l + 1
        b_q = 2 * a_l * b_l
        c_q = b2_l - r2

        d = b_q * b_q - 4 * a_q * c_q

        dist = self.laser_max_range

        if d > 0:
            y1 = (-b_q - math.sqrt(d)) / (2 * a_q)
            y2 = (-b_q + math.sqrt(d)) / (2 * a_q)

            x1 = a_l * y1 + b_l
            x2 = a_l * y2 + b_l

            dist = self.calculate_distance(np.array([x1, y1]), pos)

            dist = min(dist, self.calculate_distance(np.array([x2, y2]), pos))


        elif d == 0:
            y1 = - b_q / (2 * a_q)
            x1 = a_l * y1 + b_l
            self.calculate_distance(np.array([x1, y1]), pos)

        return dist

    def calculate_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1]))

    def generate_new_trees(self):
        return
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

    def get_obs(self):
        pass

    def computeReward(self):
        dist_margin = 0.1
        for i in range(self.current_grid[0] - 1, self.current_grid[0] + 1 + 1):
            for j in range(self.current_grid[1] - 1, self.current_grid[1] + 1 + 1):
                key = str([i, j])
                if key in self.trees:
                    for tree in self.trees[key]:
                        dist = tree.distance(self.drone.pos)

                        if dist - tree.r - dist_margin < 0:
                            print("ccolision")


    def step(self, actions):
        self.drone.make_step(actions[0], actions[1])
        self.current_grid = [int(self.drone.pos[0] / self.grid_size), int(self.drone.pos[1] / self.grid_size)]

        if self.current_grid != self.last_grid:
            self.generate_new_trees()

        self.calculate_laser_distances()

        self.computeReward()

        if self.do_render:
            self.render()

        self.last_grid = self.current_grid

    def render(self):
        start_time = time.time()
        background = np.zeros((self.window_size[0], self.window_size[1], 3), np.uint8)
        background[:] = (0, 255, 0)

        for i in range(self.current_grid[0] - 1, self.current_grid[0] + 1 + 1):
            for j in range(self.current_grid[1] - 1, self.current_grid[1] + 1 + 1):
                key = str([i, j])
                if key in self.trees:
                    for tree in self.trees[key]:
                        pos_diff = (tree.p - self.drone.pos) * self.pixels_per_meter

                        pos_diff = np.array([500 + pos_diff[1], 500 - pos_diff[0]]).astype(np.int)

                        radius = int(tree.r * self.pixels_per_meter)

                        cv2.circle(background, pos_diff, radius, (60, 103, 155), -1)

        for i in range(self.laser_resolution):
            A = int(self.laser_ranges[i] * self.pixels_per_meter * math.sin(i * self.laser_angle_per_step)) + self.mid_point[0]
            B = -int(self.laser_ranges[i] * self.pixels_per_meter * math.cos(i * self.laser_angle_per_step)) + self.mid_point[1]
            cv2.line(background, self.mid_point, (A, B), (0, 0, 255), 1)
            cv2.circle(background, (A, B), 2, (0, 0, 255), -1)

        background = cv2.rectangle(background, (490, 485), (510, 515), (0, 0, 0), -1)
        background = cv2.rectangle(background, (490, 485), (510, 490), (255, 0, 0), -1)



        cv2.imshow("game", background)
        cv2.waitKey(1)

        wait_time = self.step_time - (time.time() - start_time)
        if wait_time > 0:
            time.sleep(wait_time)
