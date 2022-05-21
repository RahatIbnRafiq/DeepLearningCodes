import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Rescaling, Activation
from keras.datasets import cifar10
import sys


def my_model():
    inputs = keras.Input(shape=(32, 32, 3))

    x = Conv2D(32, 3)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    for channel_size in [64, 128]:
        x = Conv2D(channel_size, 3)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs)


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)

model = my_model()
print(model.summary())

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

# model.evaluate(x_test, y_test, batch_size=200, verbose=2)