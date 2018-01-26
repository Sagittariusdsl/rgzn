__author__ = 'DSL'
# -*-coding:utf-8-*-
from matplotlib.colors import ListedColormap
from test import Perceptron
import numpy as np

def plot_decision_regions(x, y, classifier, resolution = 0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap   = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max()
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max()
    print(x1_min, x1_max)
    print(x2_min, x2_max)