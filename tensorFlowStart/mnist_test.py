import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.layers as layers
import os
import matplotlib.pyplot as plt

# 忽略警告及以下消息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Mnist:
    """
    mnist基类
    """
    def __init__(self, learning_rate=1e-4):
        # 加载数据集
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # 占位符
        self.x = tf.placeholder(tf.float32, [None, 784], name='X')
        self.y = tf.placeholder(tf.float32, [None, 10], name='Y')

        # 测试精度
        self.train_acc_list = []
        self.test_acc_list = []
        self.loss_list = []

    def optimizer(self, fun, learning_rate=0.01, **kwargs):
        """
        优化器
        :param fun: 优化函数
        :param loss: 损失值
        :param learning_rate: 学习速率
        :param kwargs:
        :return:
        """
        return fun(learning_rate=learning_rate, *kwargs).minimize(loss=self._loss)

    def cross_entropy_loss(self):
        # return -tf.reduce_mean(self.y * tf.log(tf.nn.softmax(self.y_pre)))

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_pre))

    def _test(self, sess, x_, y_, dropout=0.):
        feed_dict = {self.x: x_, self.y: y_}
        if dropout:
            feed_dict[self.keep_prob] = dropout
        return sess.run(self.acc, feed_dict=feed_dict)

    def test_acc(self, sess, batch_size: int = 100, dropout=0.):
        tt_batch_num = self.mnist.test.labels.shape[0] // batch_size
        test_acc = 0

        for i in range(tt_batch_num):
            test_acc += self._test(sess, self.mnist.test.images[i*batch_size: (i+1)*batch_size],
                                   self.mnist.test.labels[i*batch_size: (i+1)*batch_size], dropout)

        self.test_acc_list.append(test_acc/tt_batch_num)

    def accuracy(self):
        acc = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_pre, 1))
        return tf.reduce_mean(tf.cast(acc, tf.float32))

    def show(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_list)
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_list, '-b')
        plt.plot(self.test_acc_list, '-r')
        plt.legend(['train', 'test'])
        plt.show()

    def predict(self):
        raise ImportError

    def train(self):
        raise ImportError


class LogisticRegression(Mnist):
    """
    逻辑回归
    """
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        # 权重变量
        self.w = tf.Variable(tf.truncated_normal([784, 10]), name='W')
        self.b = tf.Variable(tf.constant(0.1, shape=[1, 10]), name='b')

        # init
        self.y_pre = self.predict()
        self._loss = self.cross_entropy_loss()
        self.train_step = self.optimizer(tf.train.AdamOptimizer, learning_rate=learning_rate)
        self.acc = self.accuracy()

    def predict(self):
        return tf.matmul(self.x, self.w) + self.b

    def train(self, iters_num: int = 10000, batch_size=100, epoch: int = 0) -> float:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iters_num):
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                loss_val,  _ = sess.run([self._loss, self.train_step], feed_dict={self.x: batch_xs, self.y: batch_ys})
                print(i + 1, ':loss=', loss_val, end='  ')
                if epoch and (i + 1) % epoch == 0:
                    acc = self._test(sess, batch_xs, batch_ys)
                    self.train_acc_list.append(acc)
                    self.test_acc(sess)
                    print('accuracy=', acc, end='')
                print()
                self.loss_list.append(loss_val)


class MLP(Mnist):
    """
    全连接层
    """
    def __init__(self, hidden_size=(100,), learning_rate=1e-4):
        """
        可以构建多个全连接层
        :param hidden_size: 每层节点元组
        :param learning_rate:
        """
        super().__init__()
        self.hidden_size = hidden_size

        # init
        self.y_pre = self.predict()
        self._loss = self.cross_entropy_loss()
        self.train_step = self.optimizer(tf.train.AdamOptimizer, learning_rate=learning_rate)
        self.acc = self.accuracy()

    def predict(self):
        layers_weights = [self.x]
        for i, units in enumerate(self.hidden_size):
            layers_weights.append(layers.fully_connected(layers_weights[i], units, activation_fn=tf.nn.relu,
                                                         scope='layer{}'.format(i+1)))
        return layers.fully_connected(layers_weights[-1], 10, activation_fn=None, scope='out')

    def train(self, iters_num: int = 10000, batch_size=100, epoch: int = 0) -> float:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iters_num):
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                loss_val,  _ = sess.run([self._loss, self.train_step], feed_dict={self.x: batch_xs, self.y: batch_ys})
                print(i + 1, ':loss=', loss_val, end='  ')
                if epoch and (i + 1) % epoch == 0:
                    acc = self._test(sess, batch_xs, batch_ys)
                    self.train_acc_list.append(acc)
                    self.test_acc(sess)
                    print('accuracy=', acc, end='')
                print()
                self.loss_list.append(loss_val)


class ConvNet(Mnist):
    """
    卷积神经网络
    两卷积+池化+两全连接+dropout
    """
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.keep_prob = tf.placeholder(tf.float32)
        self.weights = {
            # 5*5 conv, 1 inputs and 32 outputs
            'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
            # 5*5 conv, 32 inputs and 64 outputs
            'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
            'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1)),
            'out': tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        }
        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([10]))
        }

        # init
        self.y_pre = self.predict()
        self._loss = self.cross_entropy_loss()
        self.train_step = self.optimizer(tf.train.AdamOptimizer, learning_rate=learning_rate)
        self.acc = self.accuracy()

    def conv2d(self, x, w, b, strides=1):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def predict(self):
        # 将数据转换成28*28个像素点的图像
        x = tf.reshape(self.x, shape=[-1, 28, 28, 1])

        # First convolution layer
        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
        # Max pooling
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolution layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # reshape成与全连接层输入相关, 最后有64个输出，28 * 28 * 64，两个池化层减小为1/16， 故为 7 * 7 * 64
        fc1 = tf.reshape(conv2, shape=[-1, 7 * 7 * 64])

        # fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        # fc1 = tf.nn.relu(fc1)

        fc1 = layers.fully_connected(fc1, 1024, activation_fn=tf.nn.relu)

        # dropout
        fc1 = tf.nn.dropout(fc1, self.keep_prob)

        # return tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        return layers.fully_connected(fc1, 10, activation_fn=None, scope='out')

    def train(self, iters_num: int = 5000, batch_size=100, epoch: int = 0) -> float:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iters_num):
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                loss_val,  _ = sess.run([self._loss, self.train_step], feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                                  self.keep_prob: 0.8})
                print(i + 1, ':loss=', loss_val, end='  ')
                if epoch and (i + 1) % epoch == 0:
                    acc = self._test(sess, batch_xs, batch_ys, 1.)
                    self.train_acc_list.append(acc)
                    self.test_acc(sess, dropout=1.)
                    print('accuracy=', acc, end='')
                print()
                self.loss_list.append(loss_val)


if __name__ == '__main__':
    print('start....-----------.......')

    # lr = LogisticRegression()
    # lr.train(iters_num=10000, epoch=100)
    # lr.show()

    # mlp = MLP()
    # mlp.train(iters_num=10000, epoch=1000)
    # mlp.show()

    conv = ConvNet()
    conv.train(1000, epoch=100)
    conv.show()


