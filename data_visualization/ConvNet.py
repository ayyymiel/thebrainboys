import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns

action = ["Backward", "Forward", "Left", "Right"]
#bring in data

X_train=np.load("../dataset/JumbleConvNetData/X_train.npy")
y_train=np.load("../dataset/JumbleConvNetData/y_train.npy")
X_test=np.load("../dataset/JumbleConvNetData/X_test.npy")
y_test=np.load("../dataset/JumbleConvNetData/y_test.npy")
X=np.load("../dataset/JumbleConvNetData/X (1).npy")
y=np.load("../dataset/JumbleConvNetData/y.npy")

"""
X_train=np.load("Trimmed to 150/X_train.npy")
y_train=np.load("Trimmed to 150/y_train.npy")
X_test=np.load("Trimmed to 150/X_test.npy")
y_test=np.load("Trimmed to 150/y_test.npy")
X=np.load("Trimmed to 150/X.npy")
y=np.load("Trimmed to 150/y.npy")
"""

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
# 538,0.8666666746139526,128,64,32,32,64,128,512,512
# 0.3916666805744171,32,128,128,32,32,32,256,512
# 114,0.4208333194255829,128,64,32,64,32,128,512,512
# 64, 32, 64, 32, 256, 256, 256, 256
#conv layer
model.add(Conv1D(128,  1, input_shape=X_train.shape[1:])) ##32 units, kernel size, input shape
model.add(Activation('relu'))

model.add(Conv1D(64, 2))
model.add(Activation('relu'))

model.add(Conv1D(32, 1))
model.add(Activation('relu'))

model.add(Conv1D(64, 1))
model.add(Activation('relu'))

model.add(Conv1D(32, 2))
model.add(Activation('relu'))

model.add(Conv1D(128, 2))
model.add(Activation('relu'))

model.add(Conv1D(512, 2))
model.add(Activation('relu'))

model.add(Conv1D(512, 2))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(64))

model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",tf.keras.metrics.Recall()])
model.summary()

history=model.fit(X_train, y_train, batch_size=20, epochs=40,validation_data=(X_test, y_test))

model.save("CNN1.h5")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


#Confusion Matrix
pred=model.predict(X_test)
#pred=np.array(pred)
conf=confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
print('Confusion Matrix')
print (conf)
confNorm = conf/conf.astype(np.float).sum(axis=1)
print('Normalized Confusion Matrix')
print(confNorm)

df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))
plt.title('Normalized Confusion Matrix \n ConvNet')

sns.heatmap(df_cm,  annot=True)
plt.xlabel("Predicted Labels", labelpad=18)
plt.ylabel("Expected Labels",labelpad=18 )
plt.show()


