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
import time

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
# print(X_test.shape)

# normalize data
tf.keras.utils.normalize(X_train, axis=-1, order=2)  # L2 norm

# # randomize filters
# filter_list = [32, 64, 128, 256, 512]
# combinations, neurons = [], []
#
# for i in range(625):
#     random_zero = random.randint(0, 4)
#     random_one = random.randint(0, 4)
#     random_two = random.randint(0, 4)
#     random_three = random.randint(0, 4)
#     random_four = random.randint(0, 4)
#     random_five = random.randint(0, 4)
#     random_six = random.randint(0, 4)
#     random_seven = random.randint(0, 4)
#     random_eight = random.randint(0, 4)
#
#     neurons = [filter_list[random_zero], filter_list[random_one],
#                filter_list[random_two], filter_list[random_three], filter_list[random_four],
#                filter_list[random_five], filter_list[random_six], filter_list[random_seven],
#                filter_list[random_eight]]
#     if neurons in combinations:  # if the combination of neurons exists in the combinations list
#         print("Combination already exists.")
#     else:
#         combinations.append(neurons)
#
# accuracies, l1, l2, l3, l4, l5, l6, l7, l8 = [], [], [], [], [], [], [], [], []
# initialize model

combinations = pd.read_csv('new_acc.csv')
combinations = pd.DataFrame.to_numpy(combinations)
combinations = np.ndarray.tolist(combinations)
print(len(combinations))
# for i in range(len(combinations)):
#
#     model = Sequential()
#
#     model.add(Conv1D(combinations[i][0], 1, input_shape=(8, 700)))  # 32 units, kernel size, input shape
#     model.add(Activation('relu'))
#
#     model.add(Conv1D(combinations[i][1], 2))
#     model.add(Activation('relu'))
#
#     model.add(Conv1D(combinations[i][2], 1))
#     model.add(Activation('relu'))
#
#     model.add(Conv1D(combinations[i][3], 1))
#     model.add(Activation('relu'))
#
#     model.add(Conv1D(combinations[i][4], 2))
#     model.add(Activation('relu'))
#
#     model.add(Conv1D(combinations[i][5], 2))
#     model.add(Activation('relu'))
#
#     model.add(Conv1D(combinations[i][6], 2))
#     model.add(Activation('relu'))
#
#     model.add(Conv1D(combinations[i][7], 2))
#     model.add(Activation('relu'))
#
#     model.add(Flatten())
#
#     model.add(Dense(512))
#     model.add(Dense(256))
#     model.add(Dense(64))
#
#     model.add(Dense(5))
#     model.add(Activation('softmax'))
#
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", tf.keras.metrics.Recall()])
#     model.summary()
#
#     history = model.fit(X_train, y_train, batch_size=20, epochs=40, validation_data=(X_test, y_test))
#     accuracies.append(history.history['accuracy'][39])
#     l1.append(combinations[i][0])
#     l2.append(combinations[i][1])
#     l3.append(combinations[i][2])
#     l4.append(combinations[i][3])
#     l5.append(combinations[i][4])
#     l6.append(combinations[i][5])
#     l7.append(combinations[i][6])
#     l8.append(combinations[i][7])
#     print(f'Iteration {i}')
#     time.sleep(2)
#
# df = {'Accuracy': accuracies,
#       'Layer1': l1,
#       'Layer2': l2,
#       'Layer3': l3,
#       'Layer4': l4,
#       'Layer5': l5,
#       'Layer6': l6,
#       'Layer7': l7,
#       'Layer8': l8,
#       }
#
# df = pd.DataFrame(df)
# df.to_csv('accuracy.csv', encoding='utf-8', index=False, header=False)
# print('Finished randomizing. Accuracies and neurons appended')

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


