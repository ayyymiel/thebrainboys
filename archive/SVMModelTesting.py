# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split
import _pickle as cPickle
from sklearn.metrics import confusion_matrix


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


# Create a classifier: a support vector classifier
#clf = svm.SVC(  kernel='linear', C=75, gamma=0.1)
#clf = svm.SVC(  kernel='rbf', C=75, gamma=0.001) #this gave pretty good results *****************
#clf = svm.SVC(  kernel='rbf', C=90, gamma=0.0025) #might be good, but could be overfit

#clf = svm.SVC( kernel='linear', C=75, gamma=0.0001)# pretty trash
#clf = svm.SVC( kernel='poly', C=50, gamma=0.003)#

#best from grid search {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001) #this gave pretty good results
#best {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")



# Predict the value of the digit on the test subset
predicted2 = clf.predict(X_train)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_train, predicted2)}\n")

# save the classifier
with open('SVMModel.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)

#pred=np.array(pred)
conf=confusion_matrix(y_test, predicted)
print('Confusion Matrix')
print (conf)
confNorm = conf/conf.astype(np.float).sum(axis=1)
print('Normalized Confusion Matrix')
print(confNorm)

