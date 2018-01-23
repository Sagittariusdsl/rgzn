__author__ = 'DSL'
# -*-coding:utf-8-*-

import numpy as np

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
        self.w_ = np.zero(1 + x.shape[1])
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
        pass