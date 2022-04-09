from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import os
from copy import deepcopy


class SimulatedAnnealing(ABC):
    init_features : list = None
    n_iterations : int = None
    T : int = None
    save_file_dir : str = None
    save_file_path_base : str = None
    # cost, features - temporary fields

    @abstractmethod
    def __init__(self):
        if not os.path.exists(self.save_file_dir):
            os.makedirs(self.save_file_dir)

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

    def save_frame(self, frame_name):
        plt.title(f'cost: {self.cost}', size=14)
        plt.axis('off')
        plt.savefig(self.get_frame_path(frame_name))
        plt.close()

    def get_frame_path(self, frame_name):
        return f'{self.save_file_path_base}_{frame_name}.png'

    def create_init_min_imgs(self):
        self.cost = self.init_cost
        self.features = self.init_features
        self.create_frame('initial')
        self.cost = self.min_cost
        self.features = self.min_features
        self.create_frame('minimal')

    def show_init_min_imgs(self, figsize):
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path('initial')))
        ax.set_title('Initial', size=9)
        plt.axis('off')

        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path('minimal')))
        ax.set_title('Minimal', size=9)
        plt.axis('off')

        plt.show()

    def create_gif(self):
        with imageio.get_writer(f'{self.save_file_path_base}_animation.gif', mode='I') as writer:
            for i in range(-1, self.n_iterations):
                img = imageio.imread(self.get_frame_path(i))
                writer.append_data(img)

    def remove_frames(self):
        for i in range(-1, self.n_iterations):
            os.remove(self.get_frame_path(i))

    def get_acceptance_probability(self, i, old_cost, new_cost):
        return np.exp(-i*(old_cost-new_cost)/old_cost)

    def perform(self, init_min_imgs=True, gif=False):
        self.features = deepcopy(self.init_features)
        self.min_features = deepcopy(self.features)
        self.init_cost = self.min_cost = self.cost = old_cost = self.get_cost()
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

        if gif:
            self.create_gif()
        if init_min_imgs:
            self.create_init_min_imgs()
        self.remove_frames()
