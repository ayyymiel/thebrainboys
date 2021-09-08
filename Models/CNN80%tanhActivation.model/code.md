# 80% Accuracy ConvNet
##  tanh Activation function

### Model Code

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D

import numpy as np

import pickle

import matplotlib.pyplot as plt

#bring in data

X_train=np.load("X_train.npy")

y_train=np.load("y_train.npy")

X_test=np.load("X_test.npy")

y_test=np.load("y_test.npy")

X=np.load("X.npy")

y=np.load("y.npy")

X=np.array(X)

y=np.array(y)

X_train=np.array(X_train)

y_train=np.array(y_train)

X_test=np.array(X_test)

y_test=np.array(y_test)

#normalize data

tf.keras.utils.normalize(X_train, axis=-1, order=2) #L2 norm


#initialize model

model = Sequential()

#conv layer

model.add(Conv1D(64, 2,  input_shape=X_train.shape[1:])) ##32 units, kernel size, input shape

model.add(Activation('tanh'))

model.add(Conv1D(64, 2))

model.add(Activation('tanh'))

model.add(Conv1D(128, 2))

model.add(Activation('tanh'))

model.add(Conv1D(256, 1))

model.add(Activation('tanh'))

model.add(Conv1D(512, 1))

model.add(Activation('tanh'))

model.add(Conv1D(256, 2))

model.add(Activation('tanh'))

model.add(Conv1D(128, 1))

model.add(Activation('tanh'))

#model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 1))

model.add(Activation('tanh'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(128))

model.add(Dense(64))

model.add(Dense(32))


model.add(Dense(5))

model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history=model.fit(X_train, y_train, batch_size=5, epochs=50,validation_data=(X_test, y_test))

model.save("CNN1.model")

# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

"""

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()


"""
