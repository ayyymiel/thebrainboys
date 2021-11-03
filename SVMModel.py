# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split
import _pickle as cPickle
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import joblib
action = ["Backward", "Forward", "Left", "Other", "Right"]

"""
X=np.load("TrainTestDataFlat/XSVM.npy")
y=np.load("TrainTestDataFlat/ySVM.npy")
X_train=np.load("TrainTestDataFlat/X_trainSVM.npy")
y_train=np.load("TrainTestDataFlat/y_trainSVM.npy")
X_test=np.load("TrainTestDataFlat/X_testSVM.npy")
y_test=np.load("TrainTestDataFlat/y_testSVM.npy")
"""
X=np.load("TrainTestDataFlat/Shuffled/X.npy")
y=np.load("TrainTestDataFlat/Shuffled/y.npy")
X_train=np.load("TrainTestDataFlat/Shuffled/X_train.npy")
y_train=np.load("TrainTestDataFlat/Shuffled/y_train.npy")
X_test=np.load("TrainTestDataFlat/Shuffled/X_test.npy")
y_test=np.load("TrainTestDataFlat/Shuffled/y_test.npy")

X=np.array(X)
y=np.array(y)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)


# Create a classifier: a support vector classifier
#model = svm.SVC(  kernel='linear', C=75, gamma=0.1)
#model = svm.SVC(  kernel='rbf', C=75, gamma=0.001) #this gave pretty good results *****************
#model = svm.SVC(  kernel='rbf', C=90, gamma=0.0025) #might be good, but could be overfit
#model = svm.SVC( kernel='linear', C=75, gamma=0.0001)# pretty trash
#model = svm.SVC( kernel='poly', C=50, gamma=0.003)#
#best from grid search {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
model = svm.SVC(kernel='rbf', C=1000, gamma=0.001) #this gave pretty good results
#best {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
# Learn the digits on the train subset
model.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = model.predict(X_test)

print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")



# Predict the value of the digit on the test subset
predicted2 = model.predict(X_train)

print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_train, predicted2)}\n")

#save model
joblib.dump(model, "SVMModel.joblib")

"""
#to load model 
# load, no need to initialize the loaded_rf
model = joblib.load("SVMModel.joblib")
"""
#pred=np.array(pred)
conf=confusion_matrix(y_test, predicted)
print('Confusion Matrix')
print (conf)
confNorm = conf/conf.astype(np.float).sum(axis=1)
print('Normalized Confusion Matrix ')
print(confNorm)

df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))
plt.title('Normalized Confusion Matrix \n Support Vector Machine')
sns.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Labels", labelpad=18)
plt.ylabel("Expected Labels",labelpad=18 )
plt.show()

np.fill_diagonal(confNorm, 0)
df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))

plt.title('Absolute Confusion Matrix \n Support Vector Machine')
sns.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Labels", labelpad=18)
plt.ylabel("Expected Labels",labelpad=18 )
plt.show()