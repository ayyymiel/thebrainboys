# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#import data
X=np.load("XSVM.npy")

y=np.load("ySVM.npy")

X=np.array(X)
y=np.array(y)

print (y[0])

X_train=np.load("X_trainSVM.npy")
y_train=np.load("y_trainSVM.npy")
X_test=np.load("X_testSVM.npy")
y_test=np.load("y_testSVM.npy")
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]}]

scores = ['precision', 'recall']
#{'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'} best
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        svm.SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)


    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

