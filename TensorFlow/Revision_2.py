import tensorflow as tf
from tensorflow import keras
import numpy as np

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if((logs.get('accuracy'))>0.6):
        #if(logs.get('acc')>0.6):
            print("\nAccuracy of 60% reached so cancelling further training. ")
            self.model.stop_training = True

mnist = keras.datasets.fashion_mnist

(x_tr, y_tr), (x_tst, y_tst) = mnist.load_data()

x_tr, x_tst = x_tr/255.0, x_tst/255.0

callbacks = myCallback()

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_tr, y_tr, epochs = 5, callbacks = [callbacks])


