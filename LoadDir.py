import numpy as np
import matplotlib.pyplot as plt
import os
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import random
import pickle

dir = 'C:/Users/Eric/PycharmProjects/pythonProject/ActionData'
categories = ["left", "none", "right"]
#count = 0;

trainingData=[]

def createTrainingData():
    for category in categories:
        path = os.path.join(dir, category)  # joins the path to the 3 actions
        classNum=categories.index(category) #left=0, none=1, right=2

        for datas in os.listdir(path):
            actionData = DataFilter.read_file(os.path.join(path, datas))
            if(categories.index(category)==0):
                trainingData.append([actionData, [1, 0, 0]])

            elif(categories.index(category)==1):
                trainingData.append([actionData, [0, 1, 0]])

            elif(categories.index(category)==2):
                trainingData.append([actionData, [0, 0, 1]])


createTrainingData()
print(len(trainingData))

random.shuffle(trainingData) #randomize the training data

X=[] #feature
y=[] #label

for features, label in trainingData:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1, 8,700)

np.save("X.npy",X)
np.save("y.npy",y )

