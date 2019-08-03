import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
x_valid = mnist.validation.images
y_valid = mnist.validation.labels


class MLP:
    def __init__(self):

        # 层堆叠
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(100, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
                           loss='categorical_crossentropy', metrics=['accuracy'])

        # Restore the model's state,
        # this requires a model with the same architecture.
        # self.model.load_weights('./weights/mlp_model')

        # Restore the model's state
        model.load_weights('my_model.h5')

    def train(self, dataset=None):
        if not dataset:
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).repeat()
        v_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32).repeat()
        # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
        self.model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=v_dataset, validation_steps=3)

        # Save weights to a TensorFlow Checkpoint file
        # self.model.save_weights('./weights/mlp_model')

        # Save weights to a HDF5 file
        self.model.save_weights('./weights/klp_model.h5', save_format='h5')

    def predict(self, data=None):
        if not data:
            data = x_test
        return np.argmax(self.model.predict(data, batch_size=32), 1)

    def evaluate(self, dataset=None):
        if not dataset:
            dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32).repeat()

        self.model.evaluate(dataset, steps=30)


if __name__ == '__main__':
    model = MLP()
    # model.train()
