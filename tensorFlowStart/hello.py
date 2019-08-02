import tensorflow as tf
import os

# 忽略警告及以下消息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    message = tf.constant("hello world")
    with tf.compat.v1.Session() as sess:
        print(sess.run(message).decode())
