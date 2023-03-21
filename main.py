import tensorflow.keras as tfk
#from tfk.models import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Normalization
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()
mm=Sequential()
mm.add(Flatten(input_shape=(28, 28))) # flatten the input image
mm.add(Normalization()) # apply feature-wise normalization
mm.add(Dense(45, activation='relu'))
optimisation=tfk.optimizers.Adam()
with tf.GradientTape() as tape:
    y_pred=mm(x_train[1])
    bss=crossentropy(Y_train[i],Y_pred)
    grad=Tape.gradient(loss,mm.trainable_variables)
optimisateur.apply-gradient(zip(grad,mm.trainable_variables))
plt.imshow(x_train[0])
plt.show()