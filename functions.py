import numpy as np


class Function(object):

    def sigmoid(self, x):
        """y=1/(1+exp(-x))"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    def relu(self, x):
        """y=x if x>0 else 0"""
        return np.maximum(0, x)

    def relu_grad(self, x):
        grad = np.zeros(x)
        grad[x >= 0] = 1
        return grad

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))

    def mean_squared_error(self, y, t):
        """均方误差
        y:预测值；t:真实值
        :returns 误差矩阵"""
        return 0.5 * np.sum((y - t) ** 2)

    def cross_entropy_error(self, y, t):
        """交叉熵误差
        :return """
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def softmax_loss(self, X, t):
        y = self.softmax(X)
        return self.cross_entropy_error(y, t)

    def numerical_gradient(self, f, x):
        """求梯度"""
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])  # 多重索引 迭代器可读可写
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = float(tmp_val) - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 还原值
            it.iternext()

        return grad

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """

        Parameters
        ----------
        input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
        filter_h : 滤波器的高
        filter_w : 滤波器的长
        stride : 步幅
        pad : 填充

        Returns
        -------
        col : 2维数组
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col

    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """

        Parameters
        ----------
        col :
        input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
        filter_h :
        filter_w
        stride
        pad

        Returns
        -------

        """
        N, C, H, W = input_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]


