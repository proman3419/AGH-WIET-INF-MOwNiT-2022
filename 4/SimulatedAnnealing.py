from abc import ABC, abstractmethod
import numpy as np
import imageio
import time
import os
from copy import deepcopy


class SimulatedAnnealing(ABC):
    n_iterations : int = None
    T : int = None
    features : list = None

    @abstractmethod
    def get_next_T(self):
        raise NotImplementedError

    @abstractmethod
    def features_change(self):
        raise NotImplementedError

    @abstractmethod
    def reverse_features_change(self, change):
        raise NotImplementedError

    @abstractmethod
    def get_cost(self):
        raise NotImplementedError

    @abstractmethod
    def create_frame(self, frame_name):
        pass

    def create_init_min_imgs(self, timestamp):
        self.cost = self.init_cost
        self.features = self.init_features
        self.create_frame(f'{timestamp}initial')
        self.cost = self.min_cost
        self.features = self.min_features
        self.create_frame(f'{timestamp}minimal')

    def create_gif(self, timestamp):
        with imageio.get_writer(f'{timestamp}animation.gif', mode='I') as writer:
            for i in range(-1, self.n_iterations):
                img = imageio.imread(f'{i}.png')
                writer.append_data(img)

    def remove_frames(self):
        for i in range(-1, self.n_iterations):
            os.remove(f'{i}.png')

    def get_acceptance_probability(self, i, old_cost, new_cost):
        return np.exp(-i*(old_cost-new_cost)/old_cost)

    def perform(self, init_min_imgs=True, gif=False):
        self.init_cost = self.min_cost = self.cost = old_cost = self.get_cost()
        self.init_features = deepcopy(self.features)
        self.min_features = deepcopy(self.features)
        self.create_frame(-1)

        for i in range(self.n_iterations):
            self.T = self.get_next_T()
            change = self.features_change()
            self.cost = self.get_cost()

            if self.cost < self.min_cost:
                self.min_cost = self.cost
                self.min_features = deepcopy(self.features)
            else:
                x = np.random.uniform()
                if x > self.get_acceptance_probability(i, old_cost, self.cost):                    
                    self.reverse_features_change(change)

            self.create_frame(i)
            old_cost = self.cost

        timestamp = time.time()
        if gif:
            self.create_gif(timestamp)
        if init_min_imgs:
            self.create_init_min_imgs(timestamp)
        self.remove_frames()
