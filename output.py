__author__ = 'DSL'
# -*-coding:utf-8-*-
import numpy as np

class AdalineGD(object):
    """
    eta : float, 学习效率，处于0和1
    n_iter : int, 对训练数据进行学习改进次数
    w_ : 一维向量，存储权重数值
    errors_:存储每次迭代改进时，网络对数据进行错误判断的次数
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        """
        :param x: 输入样本向量
                    x:shape[n_samples, n_features]
                    x:[[1, 2, 3], [4, 5, 6]]
                    n_samples:2
                    n_features:3
        :param y: 对应样本分类  [1, -1]
        :return:
        """
        self.w_ = np.zeros(1 + x.shape[1])
        self.const_ = []
        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = y - output
            self.w[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.num()
            cost = (errors ** 2).sum()/2.0
            self.const_.append(cost)
        return self

    def net_input(self, x):
        return  np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return self.net_input(x)

    def predict(self, x):
        return np.where(self.activation(x) >= 0, 1, -1)

if __name__ == '__main__':
    print(132)