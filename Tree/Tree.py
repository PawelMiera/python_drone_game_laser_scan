import math
class Tree:
    def __init__(self, pos, radius):
        self.p = pos
        self.r = radius

    def distance(self, p2):
        return math.sqrt((p2[0] - self.p[0])*(p2[0] - self.p[0]) + (p2[1] - self.p[1])*(p2[1] - self.p[1]))

    def __str__(self):
        return str(self.p) + " " + str(self.r)