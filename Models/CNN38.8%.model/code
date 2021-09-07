#normalize data
tf.keras.utils.normalize(X, axis=-1, order=2) #L2 norm


#initialize model
model = Sequential()

#conv layer
model.add(Conv1D(64, 2,  input_shape=X.shape[1:])) ##32 units, kernel size, input shape
model.add(Activation('relu'))

model.add(Conv1D(64, 2))
model.add(Activation('relu'))

model.add(Conv1D(128, 2))
model.add(Activation('relu'))

model.add(Conv1D(256, 2))
model.add(Activation('relu'))

model.add(Conv1D(128, 1))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 1))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))


model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
