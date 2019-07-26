import numpy as np
from functions import Function
from dataset.mnist import load_mnist
from layers import Affine, SoftmaxWithLoss, Relu
import matplotlib.pyplot as plt
from collections import OrderedDict  # 有序字典

fun = Function()


class DeepLearn:

    def __init__(self):
        (self.x_train, self.t_train), (self.x_test, self.t_test) = load_mnist(flatten=True, normalize=True,
                                                                              one_hot_label=True)
        self.train_acc_list = []
        self.test_acc_list = []

    def predict(self, x):
        raise NotImplemented

    def cross_entropy_loss(self, x, t):
        y = self.predict(x)
        return fun.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])


class TwoLayerNet(DeepLearn):
    def __init__(self, hidden_size, weight_init_std=0.01):
        super().__init__()
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(self.x_train.shape[1], hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, self.t_train.shape[1])
        self.params['b2'] = np.zeros(self.t_train.shape[1])

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # w1, w2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
        #
        # a1 = np.dot(x, w1) + b1
        # z1 = fun.sigmoid(a1)
        # a2 = np.dot(z1, w2) + b2
        # return fun.softmax(a2)

        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def cross_entropy_loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def numerical_gradient(self, x, t):
        """数值微分求梯度，速度慢，有误差，但实现简单"""
        loss_w = lambda _: self.cross_entropy_loss(x, t)

        grads = dict()
        grads['W1'] = fun.numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = fun.numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = fun.numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = fun.numerical_gradient(loss_w, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """误差反向传播求梯度，用的是解析式求微分，速度快"""
        # forward
        self.cross_entropy_loss(x, t)

        # backward
        dout = self.last_layer.backward()

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = dict()
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

    def test(self):
        self.train_acc_list.append(self.accuracy(self.x_train, self.t_train))
        self.test_acc_list.append(self.accuracy(self.x_test, self.t_test))

    def start(self, iters_num=10000, batch_size=100, learning_rate=0.1, epoch=0, record=False, numerical=False):
        for i in range(iters_num):
            print('learn:', i)
            batch_mask = np.random.choice(self.x_train.shape[0], batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            print('开始计算梯度')
            if numerical:
                grad = self.numerical_gradient(x_batch, t_batch)
            else:
                grad = self.gradient(x_batch, t_batch)

            print('重新调整权重系数')
            for k in ('W1', 'b1', 'W2', 'b2'):
                self.params[k] -= learning_rate * grad[k]

            if epoch and (i+1) % epoch == 0:
                self.test()
            if record:
                loss = self.cross_entropy_loss(x_batch, t_batch)
                yield loss


if __name__ == '__main__':
    network = TwoLayerNet(100)
    train_loss = network.start(iters_num=10000, epoch=1000, record=True, numerical=False)

    # 可视化
    plt.subplot(1, 2, 1)
    plt.plot(list(train_loss))
    plt.subplot(1, 2, 2)
    plt.plot(network.train_acc_list, '-b')
    plt.plot(network.test_acc_list, '-r')
    plt.legend(['train', 'test'])

    plt.show()
    # print(DeepLearn().t_train.shape)



