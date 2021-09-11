# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split

#import data

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
#clf = svm.SVC(  kernel='rbf', C=75, gamma=0.001) #this gave pretty good results
#clf = svm.SVC(  kernel='rbf', C=90, gamma=0.0025) #might be good, but could be overfit

#clf = svm.SVC( kernel='linear', C=75, gamma=0.0001)# pretty trash
clf = svm.SVC( kernel='poly', C=50, gamma=0.003)#


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


"""

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
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

"""