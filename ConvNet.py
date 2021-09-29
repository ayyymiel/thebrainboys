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

"""
def plot_confusion_matrix(cm, names, title='Confusion matrix',
                            cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
"""

action = ["Backward", "Forward", "Left", "Other", "Right"]
#bring in data
"""
X_train=np.load("Train&TestDataConvNet/X_train.npy")
y_train=np.load("Train&TestDataConvNet/y_train.npy")
X_test=np.load("Train&TestDataConvNet/X_test.npy")
y_test=np.load("Train&TestDataConvNet/y_test.npy")
X=np.load("Train&TestDataConvNet/X.npy")
y=np.load("Train&TestDataConvNet/y.npy")
"""
X_train=np.load("Train&TestDataConvNet/Trimmed to 150/X_train.npy")
y_train=np.load("Train&TestDataConvNet/Trimmed to 150/y_train.npy")
X_test=np.load("Train&TestDataConvNet/Trimmed to 150/X_test.npy")
y_test=np.load("Train&TestDataConvNet/Trimmed to 150/y_test.npy")
X=np.load("Train&TestDataConvNet/Trimmed to 150/X.npy")
y=np.load("Train&TestDataConvNet/Trimmed to 150/y.npy")

X=np.array(X)
y=np.array(y)

X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
print (X_test.shape)
#normalize data
tf.keras.utils.normalize(X_train, axis=-1, order=2) #L2 norm


#initialize model
model = Sequential()

#conv layer
model.add(Conv1D(64,  1, input_shape=X_train.shape[1:])) ##32 units, kernel size, input shape
model.add(Activation('relu'))

model.add(Conv1D(64, 2))
model.add(Activation('relu'))

model.add(Conv1D(128, 2))
model.add(Activation('relu'))

model.add(Conv1D(256, 2))
model.add(Activation('relu'))

model.add(Conv1D(512, 2))
model.add(Activation('relu'))

model.add(Conv1D(256, 2))
model.add(Activation('relu'))

model.add(Conv1D(128, 2))
model.add(Activation('relu'))

model.add(Conv1D(128, 2))
model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

#model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
#model.add(Dense(32))
#model.add(Dense(16))


model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",tf.keras.metrics.Recall()])
model.summary()

history=model.fit(X_train, y_train, batch_size=10, epochs=20,validation_data=(X_test, y_test))

model.save("CNN1.h5")


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

#Confusion Matrix
pred=model.predict(X_test)
#pred=np.array(pred)
conf=confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
print (conf)

