from SimulatedAnnealing import SimulatedAnnealing
from BinaryImage import BinaryImage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


class SimulatedAnnealingBinaryImage(SimulatedAnnealing):
    def __init__(self, init_features, n_iterations, init_T, get_next_T_func,
                 get_cost_func, neighborhood, save_file_dir, 
                 save_file_name_base, seed=5040):
        self.init_features = init_features
        self.n_iterations = n_iterations
        self.init_T = init_T
        self.get_next_T_func = get_next_T_func
        self.get_cost_func = get_cost_func
        self.neighborhood = neighborhood
        self.save_file_dir = save_file_dir
        self.save_file_path_base = os.path.join(save_file_dir, 
                                                save_file_name_base)
        self.img_width = len(init_features[0])
        self.img_height = len(init_features)
        super().__init__(seed)

    # v SimulatedAnnealing overrides v
    def get_next_T(self, i):
        return self.get_next_T_func(self.init_T, self.T, i, self.n_iterations)

    def features_change(self):
        fti1, fti2 = np.random.randint(0, self.img_width*self.img_height, size=2)
        self.swap_features(fti1, fti2)
        return [(fti1, fti2)]

    def reverse_features_change(self, change):
        fti1, fti2 = change[0]
        self.swap_features(fti1, fti2)

    def get_cost(self, init=False, change=None):
        if init:
            cost = 0
            for i in range(self.img_width):
                for j in range(self.img_height):
                    cost += self.get_cost_func(self.features, i, j, self.img_width, self.img_height, self.neighborhood)
        else:
            fti1, fti2 = change[0]
            ftir1, ftic1 = self.extract_ids_for_feature(fti1)
            ftir2, ftic2 = self.extract_ids_for_feature(fti2)

            def calculate_subcost(fti1, fti2):
                self.swap_features(fti1, fti2)
                subcost = 0
                for nd in self.neighborhood:
                    ndi, ndj = nd
                    subcost += self.get_cost_func(self.features, ftir1+ndi, 
                                                      ftic1+ndj, 
                                                      self.img_width, self.img_height,
                                                      self.neighborhood)
                    subcost += self.get_cost_func(self.features, ftir2+ndi, 
                                                      ftic2+ndj, 
                                                      self.img_width, self.img_height,
                                                      self.neighborhood)
                return subcost

            old_subcost = calculate_subcost(fti1, fti2)
            new_subcost = calculate_subcost(fti1, fti2)
            cost = self.cost + (new_subcost - old_subcost)
        return cost

    def create_frame(self, frame_name):
        if frame_name == 'initial':
            BinaryImage.get_img(self.init_features).save(self.get_frame_path(frame_name))
        elif frame_name == 'minimal':
            BinaryImage.get_img(self.min_features).save(self.get_frame_path(frame_name))

        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(1, 1, 1)
        imgplot = plt.imshow(mpimg.imread(self.get_frame_path(frame_name)))
        plt.axis('off')

        self.save_frame(frame_name, cost_size=38)
    # ^ SimulatedAnnealing overrides ^

    def swap_features(self, fti1, fti2):
        ftir1, ftic1 = self.extract_ids_for_feature(fti1)
        ftir2, ftic2 = self.extract_ids_for_feature(fti2)
        self.features[ftir1][ftic1], self.features[ftir2][ftic2] = \
        self.features[ftir2][ftic2], self.features[ftir1][ftic1]

    def extract_ids_for_feature(self, fti):
        return fti//self.img_width, fti%self.img_height
