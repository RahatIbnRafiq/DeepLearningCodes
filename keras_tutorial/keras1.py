import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
import sys


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequential API

# model = Sequential()
# model.add(keras.Input(shape=(28*28)))
# model.add(Dense(units=512, activation='relu'))
# model.add(Dense(units=256, activation='relu'))
# model.add(Dense(units=10))
# print(model.summary())


#functional API

inputs = keras.Input(shape=(784,))
x = Dense(512, activation='relu', name='FirstDenseLayer')(inputs)
x = Dense(256, activation='relu', name='SecondDenseLayer')(x)
outputs = Dense(10, activation='softmax', name='SoftmaxLayer')(x)


model = keras.Model(inputs=inputs,
                    outputs=outputs,
                    name="mnist_model")
keras.utils.plot_model(model, "mnist_model.png", show_shapes=True)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=2, verbose=2)

layer_name = 'SecondDenseLayer'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(x_train)
print(intermediate_output[0])


model.evaluate(x_test, y_test, batch_size=100, verbose=2)



