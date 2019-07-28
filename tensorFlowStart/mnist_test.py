import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt

# 忽略警告及以下消息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Mnist:
    def __init__(self, ws=None, bs=None):
        # 加载数据集
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # 占位符
        self.x = tf.placeholder(tf.float32, [None, 784], name='X')
        self.y = tf.placeholder(tf.float32, [None, 10], name='Y')

        # 权重变量
        self.w = ws
        self.b = bs
        if not ws:
            self.w = tf.Variable(tf.zeros([784, 10]), name='W')
        if not bs:
            self.b = tf.Variable(tf.zeros([1, 10]), name='b')

    def optimizer(self, fun, loss, learning_rate=0.01, **kwargs):
        """
        优化器
        :param fun: 优化函数
        :param loss: 损失值
        :param learning_rate: 学习速率
        :param kwargs:
        :return:
        """
        return fun(learning_rate=learning_rate, *kwargs).minimize(loss=loss)

    def cross_entropy_loss(self, y_labal, y_predict):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_labal, logits=y_predict))

    def accuracy(self, y_labal, y_predict):
        acc = tf.equal(tf.argmax(y_labal, 1), tf.argmax(y_predict, 1))
        return tf.reduce_mean(tf.cast(acc, tf.float32))

    def predict(self, x, w, b):
        raise ImportError


class LogisticRegression(Mnist):
    def __init__(self):
        super().__init__()
        self.train_list = []
        self.test_list = []

    def predict(self, x, w, b):
        return tf.nn.softmax(tf.matmul(x, w) + b)

    def test(self, sess, accuracy):
        self.train_list.append(sess.run(accuracy, feed_dict={self.x: self.mnist.train.images,
                                                             self.y: self.mnist.train.labels}))
        self.test_list.append(sess.run(accuracy, feed_dict={self.x: self.mnist.test.images,
                                                            self.y: self.mnist.test.labels}))

    def train(self, iters_num=10000, learning_rate=0.01, batch_size=100, epoch=0):
        print('...start')
        y_pre = self.predict(self.x, self.w, self.b)
        _loss = self.cross_entropy_loss(self.y, y_pre)
        optimizer = self.optimizer(tf.train.GradientDescentOptimizer, _loss, learning_rate=learning_rate)
        accuracy = self.accuracy(self.y, y_pre)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(iters_num):
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                _, loss_val = sess.run([optimizer, _loss], feed_dict={self.x: batch_xs, self.y: batch_ys})
                if epoch and (i + 1) % epoch == 0:
                    self.test(sess, accuracy)
                print(i+1, ':loss=', loss_val)
                yield loss_val

    def show(self, _loss):
        plt.subplot(1, 2, 1)
        plt.plot(list(_loss))
        plt.subplot(1, 2, 2)
        plt.plot(self.train_list, '-b')
        plt.plot(self.test_list, '-r')
        plt.legend(['train', 'test'])
        plt.show()


if __name__ == '__main__':
    lr = LogisticRegression()
    loss = lr.train(epoch=100)
    lr.show(loss)


# # tensor board
# # 加载数据集
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# # 占位符
# x = tf.placeholder(tf.float32, [None, 784], name='X')
# y = tf.placeholder(tf.float32, [None, 10], name='Y')
#
# # 权重变量
# w = tf.Variable(tf.zeros([784, 10]), name='W')
# b = tf.Variable(tf.zeros([1, 10]), name='b')
#
# # 逻辑回归
# with tf.name_scope('wx_b') as scope:
#     y_hat = tf.nn.softmax(tf.matmul(x, w) + b)
#     # 收集w和b变化关系
#     w_h = tf.summary.histogram('weights', w)
#     b_h = tf.summary.histogram('biases', b)
#
# # 交叉熵和损失函数
# with tf.name_scope('cross-entropy') as scope:
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
#     tf.summary.scalar('cross-entropy', loss)
#
# # 梯度下降优化器
# with tf.name_scope('train') as scope:
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss=loss)
#
# # 变量初始化
# init = tf.global_variables_initializer()
#
# # 组合所有summary操作
# merged_summary_op = tf.summary.merge_all()
#
# # 存储summary
# with tf.Session() as sess:
#     sess.run(init)
#     summary_writer = tf.summary.FileWriter('graphs', sess.graph)
#
#     # 开始训练
#     for epoch in range(100):
#         loss_avg = 0
#         num_of_batch = int(mnist.train.num_examples/100)
#         for i in range(num_of_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(100)
#             _, l, summary_str = sess.run([optimizer, loss, merged_summary_op], feed_dict={x: batch_xs, y: batch_ys})
#             loss_avg += l
#             summary_writer.add_summary(summary_str, epoch*num_of_batch + i)
#         loss_avg = loss_avg/num_of_batch
#         print('Epoch{}: Loss{}'.format(epoch, loss_avg))
#     print('Done')
#     # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

