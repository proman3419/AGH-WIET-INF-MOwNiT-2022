import numpy as np
import matplotlib.pyplot as plt
from SimulatedAnnealing import SimulatedAnnealing
import os
from collections import defaultdict


class SimulatedAnnealingSudoku(SimulatedAnnealing):
    def __init__(self, n_iterations, init_T, get_next_T_func, 
                 sudoku_file_path, seed=5040):
        self.box_unknowns_is = [[] for _ in range(9)]
        self.boxis_with_unknowns = []
        self.sudoku_file_path = sudoku_file_path
        self.init_features = self.read_from_file(sudoku_file_path)
        self.n_iterations = n_iterations
        self.init_T = init_T
        self.get_next_T_func = get_next_T_func
        self.n = len(self.init_features)
        super().__init__(seed)

    # v SimulatedAnnealing overrides v
    def get_next_T(self, i):
        return self.get_next_T_func(self.init_T, self.T, i, self.n_iterations)

    def features_change(self):
        bi = np.random.choice(self.boxis_with_unknowns)
        # print(bi)
        fti1, fti2 = np.random.choice(self.box_unknowns_is[bi], size=2, replace=False)
        self.swap_features(bi, fti1, fti2)
        return [(bi, fti1, fti2)]

    def reverse_features_change(self, change):
        bi, fti1, fti2 = change[0]
        self.swap_features(bi, fti1, fti2)

    def get_cost(self, init=False, change=None):
        cost = 0
        for i in range(9):
            cost += self.occ_to_cost(self.get_occ_in_row(i)) + \
                    self.occ_to_cost(self.get_occ_in_col(i)) + \
                    self.occ_to_cost(self.get_occ_in_box(i))

        return cost

    def create_frame(self, frame_name):
        pass
    # ^ SimulatedAnnealing overrides ^

    def swap_features(self, bi, fti1, fti2):
        ftri1 = (bi//3)*3 + fti1//3
        ftci1 = (bi%3)*3 + fti1%3
        ftri2 = (bi//3)*3 + fti2//3
        ftci2 = (bi%3)*3 + fti2%3

        self.features[ftri1][ftci1], self.features[ftri2][ftci2] = \
        self.features[ftri2][ftci2], self.features[ftri1][ftci1]

    def occ_to_cost(self, occ):
        return np.sum([max(0, o-1) for o in occ])

    def get_occ_in_row(self, ri):
        occ = [0]*10
        for i in range(9):
            occ[self.features[ri][i]] += 1
        return occ

    def get_occ_in_col(self, ci):
        occ = [0]*10
        for i in range(9):
            occ[self.features[i][ci]] += 1
        return occ

    def get_occ_in_box(self, bi):
        occ = [0]*10
        for i in range(3):
            for j in range(3):
                occ[self.features[(bi//3)*3+i][(bi%3)*3+j]] = 1
        return occ

    def replace_unknowns(self):
        for bi in range(9):
            occ = self.get_occ_in_box(bi)
            occ_i = 1
            for i in range(3):
                for j in range(3):
                    if self.features[(bi//3)*3+i][(bi%3)*3+j] == 0:
                        self.box_unknowns_is[bi].append(i*3+j)
                        while occ[occ_i] == 1:
                            occ_i += 1
                        self.features[(bi//3)*3+i][(bi%3)*3+j] = occ_i
                        occ[occ_i] += 1
            if len(self.box_unknowns_is[bi]) > 0:
                self.boxis_with_unknowns.append(bi)
        return self.features

    def read_from_file(self, sudoku_file_path):
        features = []
        with open(sudoku_file_path, 'r') as f:
            for l in f.read().splitlines():
                if l.strip():
                    features.append([])
                    for ch in l:
                        if ch != ' ':
                            if ch == 'x':
                                ch = '0'
                            features[-1].append(int(ch))
        self.features = features
        return self.replace_unknowns()

    def __str__(self):
        sudoku_str = ''
        for i in range(9):
            if i in [3, 6]:
                sudoku_str += ' '*11 + '\n'
            for j in range(9):
                if j in [3, 6]:
                    sudoku_str += ' '
                sudoku_str += f'{self.features[i][j]}'
            if i != 8:
                sudoku_str += '\n'
        return sudoku_str

    def verify(self, sudoku_ans_file_path):
        self.read_from_file(sudoku_ans_file_path)
        correct = all([self.features[i][j] == self.min_features[i][j] for i in range(9) for j in range(9)])
        if correct:
            print('~=] ROZWIĄZANIE POPRAWNE [=~')
        else:
            print('~=] ROZWIĄZANIE NIEPOPRAWNE [=~')

    def print_input_file(self):
        print('>>> Plik wejściowy')
        with open(self.sudoku_file_path) as f:
            for l in f.readlines():
                print(l.strip())
        print()

    def solve(self, sudoku_ans_file_path=None):
        self.perform(init_min_imgs=False, gif=False)

        self.features = self.min_features
        self.cost = self.min_cost

        print('>>> Stan minimalny')
        print(self)
        print(f'Koszt: {self.get_cost()}')
        if sudoku_ans_file_path != None:
            self.verify(sudoku_ans_file_path)
