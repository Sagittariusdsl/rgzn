__author__ = 'DSL'
# -*-coding:utf-8-*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """
    eta:        学习率
    n_iter:     权重向量的训练次数
    w_:         神经分叉权重向量
    errors_:    用于记录神经元判断出错次数
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        """
        输入训练数据，培训神经元，
        :param x: 输入样本向量
                    x:shape[n_samples, n_features]
                    x:[[1, 2, 3], [4, 5, 6]]
                    n_samples:2
                    n_features:3
        :param y: 对应样本分类  [1, -1]
        :return:
        """
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] = update
                errors += int( update != 0.0 )
                self.errors_.append(errors)

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def getData(self):
        file = 'data.csv'
        df = pd.read_csv(file, header=None)
        y = df.loc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        x = df.loc[0:100, [0, 2]].values
        return x, y

    def showError(self):
        x, y = self.getData()
        self.fit(x, y)
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, marker = 'o')
        plt.xlabel('Errors')
        plt.ylabel('错误分类次数')
        plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        #print(idx, cl)
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=marker[idx], label=cl)

if __name__ == '__main__':
    ppn = Perceptron(eta=0.1, n_iter=10)
    x, y =ppn.getData()
    ppn.fit(x, y )
    plot_decision_regions(x, y, ppn, resolution=0.02)
    plt.xlabel('花瓣长度')
    plt.ylabel('花径长度')
    plt.legend(loc='upper left')
    plt.show()