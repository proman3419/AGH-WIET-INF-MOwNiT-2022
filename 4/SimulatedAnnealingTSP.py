import numpy as np
import matplotlib.pyplot as plt
from Point import Point
from SimulatedAnnealing import SimulatedAnnealing


class SimulatedAnnealingTSP(SimulatedAnnealing):
    def __init__(self, n_iterations, T, n):
        self.n_iterations = n_iterations
        self.T = T
        self.n = n
        self.features = self.generate_points(n, -10, 10, -10, 10)

    # v SimulatedAnnealing overrides v
    def get_next_T(self):
        return self.T * 0.99

    def features_change(self):
        fti1, fti2 = np.random.randint(0, self.n, size=2)
        self.swap_features(fti1, fti2)
        return [(fti1, fti2)]

    def reverse_features_change(self, change):
        fti1, fti2 = change[0]
        self.swap_features(fti1, fti2)

    def get_cost(self):
        cost = 0.0
        for i, ft in enumerate(self.features):
            cost += ft.get_dist(self.get_next_feature(i))

        return cost

    def create_frame(self, frame_name):
        for i, ft in enumerate(self.features):
            next_ft = self.get_next_feature(i)
            plt.plot([ft.x, next_ft.x], [ft.y, next_ft.y], 'g')
            plt.plot(ft.x, ft.y, 'ro')
        plt.title(f'cost: {self.cost}')
        plt.savefig(f'{frame_name}.png')
        plt.clf()
    # ^ SimulatedAnnealing overrides ^

    def generate_points(self, n, min_x, max_x, min_y, max_y):
        return np.array([Point(np.random.uniform(min_x, max_x), 
                               np.random.uniform(min_y, max_y)) 
                        for _ in range(n)])

    def get_next_feature(self, i):
        return self.features[(i+1)%self.n]

    def swap_features(self, fti1, fti2):
        self.features[fti1], self.features[fti2] = self.features[fti2], self.features[fti1]


if __name__ == '__main__':
    tsp = SimulatedAnnealingTSP(10, 100, 5)
    tsp.perform(init_min_imgs=False, gif=True)
    print(tsp.init_cost, tsp.min_cost)
