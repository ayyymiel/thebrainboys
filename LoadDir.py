import numpy as np
#import matplotlib.pyplot as plt
import os
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import random
import pickle
from sklearn.model_selection import train_test_split

#dir = 'C:/Users/Eric-/OneDrive/Desktop/Cap/pythonProject/ActionData/'
dir = 'TrimmedData/'
#categories = ["Back", "Forward", "Left", "Other", "Right"]
categories = ["Backward", "Forward", "Left", "Other", "Right"]


trainingData=[]

def createTrainingData():
    for category in categories:
        path = os.path.join(dir, category)  # joins the path to the 3 actions
        classNum = categories.index(category) #Backward=0, Forward =1, Left =2, Other =3, Right=4
        #classNum=categories.index(category) #left=0, none=1, right=2

        for datas in os.listdir(path):
            actionData = DataFilter.read_file(os.path.join(path, datas))
            if(categories.index(category)==0): #back
                trainingData.append([actionData, [1, 0, 0, 0, 0]])

            elif(categories.index(category)==1): #forward
                trainingData.append([actionData, [0, 1, 0, 0, 0]])

            elif(categories.index(category)==2): #left
                trainingData.append([actionData, [0, 0, 1, 0, 0]])

            elif(categories.index(category)==3): #other
                trainingData.append([actionData, [0, 0, 0, 1, 0]])

            elif(categories.index(category)==4): #right
                trainingData.append([actionData, [0, 0, 0, 0, 1]])


createTrainingData()
print(len(trainingData))

# random.shuffle(trainingData) #randomize the training data

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
X = np.array(X)
print(X.shape)
X=np.array(X).reshape(-1, 8,700)

#X=np.array(X).reshape(-1, 23,700)
np.save("X_b.npy",X)
# np.save("y.npy",y )
X_train, X_test = train_test_split(X, test_size=0.4)

#y=np.array(y).reshape(5)
np.save("X_train_b.npy", X_train)
# np.save("y_train.npy",y_train )

np.save("X_test_b.npy", X_test)
# np.save("y_test.npy",y_test)