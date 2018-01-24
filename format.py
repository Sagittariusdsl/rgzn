__author__ = 'DSL'
# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = 'data.csv'
df = pd.read_csv(file, header=None)
y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0,2]].values
plt.scatter(x[:50, 0], x[:50, 1], color = 'red', marker='o', label = 'setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color = 'blue', marker='x', label = 'versicolor')
plt.xlabel('花瓣长度')
plt.ylabel('花径长度')
plt.legend(loc = 'upper left')
plt.show()