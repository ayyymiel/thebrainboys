# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
import numpy as np
from sklearn.model_selection import train_test_split
import _pickle as cPickle
import xgboost as xgb
from xgboost import XGBClassifier
import pickle

X=np.load("XSVM.npy")
y=np.load("ySVM.npy")

X=np.array(X)
y=np.array(y)

X_train=np.load("X_trainSVM.npy")
y_train=np.load("y_trainSVM.npy")
X_test=np.load("X_testSVM.npy")
y_test=np.load("y_testSVM.npy")
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

clf = XGBClassifier()
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

# Predict the value of the digit on the test subset
predicted2 = clf.predict(X_train)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_train, predicted2)}\n")

file_name = "xgboost.pkl"

# save
pickle.dump(clf, open(file_name, "wb"))