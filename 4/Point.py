import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_dist(self, o):
        return np.sqrt(np.abs(self.x-o.x)**2 + np.abs(self.y-o.y)**2)

    def __sub__(self, o):
        return self.get_dist(o)
