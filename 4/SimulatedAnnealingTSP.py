import numpy as np
import matplotlib.pyplot as plt
from Point import Point
from SimulatedAnnealing import SimulatedAnnealing
import os


class SimulatedAnnealingTSP(SimulatedAnnealing):
    def __init__(self, init_features, n_iterations, init_T, get_next_T_func,
                 save_file_dir, save_file_name_base):
        self.init_features = init_features
        self.n_iterations = n_iterations
        self.init_T = init_T
        self.save_file_dir = save_file_dir
        self.save_file_path_base = os.path.join(save_file_dir, 
                                                save_file_name_base)
        self.n = len(init_features)
        super().__init__()

    # v SimulatedAnnealing overrides v
    def get_next_T(self, i):
        return get_next_T_func(self.init_T, self.T, i)

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
        self.save_frame(frame_name)
    # ^ SimulatedAnnealing overrides ^

    def get_next_feature(self, i):
        return self.features[(i+1)%self.n]

    def swap_features(self, fti1, fti2):
        self.features[fti1], self.features[fti2] = self.features[fti2], self.features[fti1]


def generate_points(n, min_x, max_x, min_y, max_y):
    return np.array([Point(np.random.uniform(min_x, max_x), 
                           np.random.uniform(min_y, max_y)) 
                    for _ in range(n)])


if __name__ == '__main__':
    np.random.seed(1337)
    init_features = generate_points(10**2, -10, 10, -10, 10)
    def get_next_T_func(init_T, T, i):
        return T * 0.99
    tsp = SimulatedAnnealingTSP(init_features, n_iterations=10**3, init_T=10**3,
                                get_next_T_func=get_next_T_func,
                                save_file_dir='output', save_file_name_base='tsp')
    tsp.perform(init_min_imgs=True, gif=False)
    tsp.show_init_min_imgs()
    tsp.show_cost_graph()
    # tsp.show_temperature_graph()
