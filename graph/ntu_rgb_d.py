import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools
import matplotlib.pyplot as plt
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]

outward = [(j, i) for (i, j) in inward]

neighbor = inward + outward


head = [(2, 3), (2, 20), (20, 4), (20, 8)]
lefthand = [(4, 5), (5, 6), (6, 7), (7, 22), (22, 21)]
righthand = [(8, 9), (9, 10), (10, 11), (11, 24), (24, 23)]
hands = lefthand + righthand
torso = [(20, 4), (20, 8), (20, 1), (1, 0), (0, 12), (0, 16)]
leftleg = [(0, 12), (12, 13), (13, 14), (14, 15)]
rightleg = [(0, 16), (16, 17), (17, 18), (18, 19)]
legs = leftleg + rightleg
parts = [head, hands, torso, legs]



class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.parts = parts
        self.A = self.get_adjacency_matrix(labeling_mode)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'parts':
            A = tools.get_part_based_graph(self.num_node, self.self_link, self.parts)
        else:
            raise ValueError()

        return A


