import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

action = ["Backward", "Forward", "Left", "Other", "Right"]
# bring in data
X_train = np.load("ConvNet Dataset/X_train.npy")
y_train = np.load("ConvNet Dataset/y_train.npy")
X_test = np.load("ConvNet Dataset/X_test.npy")
y_test = np.load("ConvNet Dataset/y_test.npy")
X = np.load("ConvNet Dataset/X.npy")
y = np.load("ConvNet Dataset/y.npy")

X = np.array(X)
y = np.array(y)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_test.shape)

# normalize data
tf.keras.utils.normalize(X_train, axis=-1, order=2)  # L2 norm


# initialize model
model = Sequential()

# randomize filters
filter_list = [32, 64, 128, 256, 512]
combinations, neurons = [], []

for i in range(625):
    random_zero = random.randint(0, 4)
    random_one = random.randint(0, 4)
    random_two = random.randint(0, 4)
    random_three = random.randint(0, 4)
    random_four = random.randint(0, 4)
    random_five = random.randint(0, 4)
    random_six = random.randint(0, 4)
    random_seven = random.randint(0, 4)
    random_eight = random.randint(0, 4)

    neurons = [filter_list[random_zero], filter_list[random_one],
               filter_list[random_two], filter_list[random_three], filter_list[random_four],
               filter_list[random_five], filter_list[random_six], filter_list[random_seven],
               filter_list[random_eight]]
    if neurons in combinations:  # if the combination of neurons exists in the combinations list
        print("Combination already exists.")
    else:
        combinations.append(neurons)

print(combinations)

#
# # conv layer
# model.add(Conv1D(128,  1, input_shape=X_train.shape[1:]))  # 32 units, kernel size, input shape
# model.add(Activation('relu'))
#
# model.add(Conv1D(128, 2))
# model.add(Activation('relu'))
#
# model.add(Conv1D(256, 1))
# model.add(Activation('relu'))
#
# model.add(Conv1D(256, 1))
# model.add(Activation('relu'))
#
# model.add(Conv1D(512, 2))
# model.add(Activation('relu'))
#
# model.add(Conv1D(256, 2))
# model.add(Activation('relu'))
#
# model.add(Conv1D(128, 2))
# model.add(Activation('relu'))
#
# model.add(Conv1D(128, 2))
# model.add(Activation('relu'))
#
# model.add(Flatten())
#
# model.add(Dense(512))
# model.add(Dense(256))
# model.add(Dense(64))
#
# model.add(Dense(5))
# model.add(Activation('softmax'))
#
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", tf.keras.metrics.Recall()])
# model.summary()
#
# history = model.fit(X_train, y_train, batch_size=20, epochs=40, validation_data=(X_test, y_test))
# accuracies.append(history.history['accuracy'][39])
#
# df = pd.DataFrame(accuracies)
# df.to_csv('accuracy.csv', encoding='utf-8', index=False, header=False)

# skip for now
"""
# model.save("CNN1.h5")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# Confusion Matrix
pred = model.predict(X_test)
conf = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
print('Confusion Matrix')
print(conf)
confNorm = conf/conf.astype(np.float).sum(axis=1)
print('Normalized Confusion Matrix')
print(confNorm)
plt.show()
"""


