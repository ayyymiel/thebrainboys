import numpy as np
#import matplotlib.pyplot as plt
import os
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#dir = 'C:/Users/Eric-/OneDrive/Desktop/Cap/pythonProject/ActionData/'
dir = 'C:/Users/Eric-/OneDrive/Desktop/Cap/pythonProject/ActionsData/'
#categories = ["Back", "Forward", "Left", "Other", "Right"]
categories = ["Backward", "Forward", "Left", "Other", "Right"]


trainingData=[]
trainingLabel=[]

def createTrainingData():
    for category in categories:
        path = os.path.join(dir, category)  # joins the path to the 3 actions
        classNum = categories.index(category) #Backward=0, Forward =1, Left =2, Other =3, Right=4
        #classNum=categories.index(category) #left=0, none=1, right=2

        for datas in os.listdir(path):
            actionData = DataFilter.read_file(os.path.join(path, datas))
            if(categories.index(category)==0): #back
                trainingData.append([actionData, 0])

            elif(categories.index(category)==1): #forward
                trainingData.append([actionData, 1])

            elif(categories.index(category)==2): #left
                trainingData.append([actionData, 2])

            elif(categories.index(category)==3): #other
                trainingData.append([actionData, 3])

            elif(categories.index(category)==4): #right
                trainingData.append([actionData, 4])


createTrainingData()
print(len(trainingData))

#random.shuffle(trainingData) #randomize the training data

X=[] #feature
y=[] #label

for features, label in trainingData:
    X.append(features)
    y.append(label)
"""
#print (X[1].shape)
for i in range(499):
    #print (X[i].shape)
    if(X[i].shape!=(8,700)):
        print(X[i].shape)
        print(y[i])
"""


X=np.array(X).reshape(-1, 8,700)
n_samples = len(X)
X = X.reshape((n_samples, -1))
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
X = scaler.fit_transform(X)
#n_samples = len(y)
#y = y.reshape((n_samples, -1))

#X=np.array(X).reshape(-1, 23,700)
np.save("XSVM.npy",X)
np.save("ySVM.npy",y )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#y=np.array(y).reshape(5)
np.save("X_trainSVM.npy",X_train)
np.save("y_trainSVM.npy",y_train )

np.save("X_testSVM.npy",X_test)
np.save("y_testSVM.npy",y_test)