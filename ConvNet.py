import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import numpy as np
import pickle

#bring in data

X=np.load("X.npy")
y=np.load("y.npy")

X=np.array(X)
y=np.array(y)

#normalize data
tf.keras.utils.normalize(X, axis=-1, order=2) #L2 norm

#initialize model
model = Sequential()

#conv layer
model.add(Conv1D(32, (3),  input_shape=X.shape[1:])) ##32 units, unit size, input shape
model.add(Activation('relu'))


model.add(Conv1D(32, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(32, (2)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(256))

model.add(Dense(3))
model.add(Activation('softmax'))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X, y, batch_size=5, epochs=5, validation_split=0.4)

model.save("CNN1.model")



