import numpy as np
import matplotlib.pyplot as plt
from Point import Point


class SimulatedAnnealingTSP:
    def __init__(self, n):
        self.n = n
        self.points = self.generate_points(n, -10, 10, -10, 10)
        self.fig = plt.figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(121)

    def generate_points(self, n, min_x, max_x, min_y, max_y):
        return np.array([Point(np.random.uniform(min_x, max_x), 
                               np.random.uniform(min_y, max_y)) 
                        for _ in range(n)])

    def get_next_point(self, i):
        return self.points[(i+1)%self.n]

    def get_total_dist(self):
        total_dist = 0.0
        for i, pt in enumerate(self.points):
            total_dist += pt.get_dist(self.get_next_point(i))

        return total_dist

    def swap_points(self, pti1, pti2):
        self.points[pti1], self.points[pti2] = self.points[pti2], self.points[pti1]

    def plot(self):
        for i, pt in enumerate(self.points):
            next_pt = self.get_next_point(i)
            self.ax.plot([pt.x, next_pt.x], [pt.y, next_pt.y], 'g')
            self.ax.plot(pt.x, pt.y, 'ro')
        plt.show()

    def solve(self):
        min_cost = cost = self.get_total_dist()
        T = 100
        T_multiplier = 0.99
        acceptance_probability = 0.8

        print('initial cost:', min_cost)

        for i in range(100):
            T *= T_multiplier
            pti1, pti2 = np.random.randint(0, self.n, size=2)
            self.swap_points(pti1, pti2)
            cost = self.get_total_dist()

            if cost < min_cost:
                min_cost = cost
            else:
                x = np.random.uniform()
                if x < acceptance_probability:
                    min_cost = cost
                else:
                    self.points[pti1], self.points[pti2] = self.points[pti2], self.points[pti1]

        print('final cost:', min_cost)


if __name__ == '__main__':
    tsp = SimulatedAnnealingTSP(20)
    tsp.plot()
    tsp.solve()
    tsp.plot()
