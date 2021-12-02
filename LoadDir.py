import numpy as np
#import matplotlib.pyplot as plt
import os
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#dir = 'C:/Users/Eric-/OneDrive/Desktop/Cap/pythonProject/ActionData/'
#dir = 'C:/Users/Eric-/OneDrive/Desktop/Cap/pythonProject/ActionsData/'
dir='NewJumbleData/'
#categories = ["Back", "Forward", "Left", "Other", "Right"]
categories = ["JumbleBackward", "JumbleForward", "JumbleLeft", "JumbleRight"]


trainingData=[]

def createTrainingData():
    for category in categories:
        path = os.path.join(dir, category)  # joins the path to the 3 actions
        classNum = categories.index(category) #Backward=0, Forward =1, Left =2, Other =3, Right=4
        #classNum=categories.index(category) #left=0, none=1, right=2

        for datas in os.listdir(path):
            actionData = DataFilter.read_file(os.path.join(path, datas))
            if(categories.index(category)==0): #back
                trainingData.append([actionData, [1, 0, 0, 0]])
                temp = actionData

            elif(categories.index(category)==1): #forward
                trainingData.append([actionData, [0, 1, 0, 0]])

            elif(categories.index(category)==2): #left
                trainingData.append([actionData, [0, 0, 1, 0]])

            elif(categories.index(category)==4): #right
                trainingData.append([actionData, [0, 0, 0, 1]])

"""
#testing if a file is too small 
if(actionData.shape!=(8,700)):
    print(datas)
"""

createTrainingData()
print(len(trainingData))

#random.shuffle(trainingData) #randomize the training data

X=[] #feature
y=[] #label

for features, label in trainingData:
    #X.append(features[:,0:150])
    X.append(features[1:9, 0:700])
    y.append(label)

"""
#print (X[1].shape)
for i in range(749):
    #print (X[i].shape)
    if(X[i].shape!=(8,700)):
        print(X[i].shape)
        print(y[i])
"""


#X=np.array(X).reshape(-1, 8,150)
X=np.array(X).reshape(-1, 8, 700)


np.save("X(new).npy", X)
np.save("y(new).npy", y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
print(X_train.shape)
print(X_test.shape)
print (X[0].shape)
#y=np.array(y).reshape(5)
np.save("X_train(new).npy", X_train)
np.save("y_train(new).npy", y_train)

np.save("X_test(new).npy", X_test)
np.save("y_test(new).npy", y_test)

"""

np.save("Train&TestDataConvNet/NewTrimmed150/X.npy", X)
np.save("Train&TestDataConvNet/NewTrimmed150/y.npy", y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,stratify=y)

print(X_train.shape)
print(X_test.shape)
print (X[0].shape)
#y=np.array(y).reshape(5)
np.save("Train&TestDataConvNet/NewTrimmed150/X_train.npy", X_train)
np.save("Train&TestDataConvNet/NewTrimmed150/y_train.npy", y_train)

np.save("Train&TestDataConvNet/NewTrimmed150/X_test.npy", X_test)
np.save("Train&TestDataConvNet/NewTrimmed150/y_test.npy", y_test)


"""