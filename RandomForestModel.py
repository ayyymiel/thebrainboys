import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sns
import _pickle as cPickle
import os
import joblib
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report›


rf = RandomForestClassifier()
action = ["Backward", "Forward", "Left", "Other", "Right"]
#import data

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

"""
{'bootstrap': True,
 'max_depth': 110,
 'max_features': 2,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 2000}
 """
rf = RandomForestClassifier(bootstrap=True, max_depth=110, max_features=2, min_samples_leaf=1, min_samples_split=2, n_estimators=2000)


# Learn the digits on the train subset
rf.fit(X_train, y_train)

#with open('RFModel.pkl', 'wb') as f:
 #   cPickle.dump(rf, f)

#save model
# joblib.dump(rf, "RFModel.joblib")

"""
#to load model 
# load, no need to initialize the loaded_rf
model = joblib.load("RFModel.joblib")
"""


# Predict the value of the digit on the test subset
predicted = rf.predict(X_test)

print(f"Classification report for classifier {rf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")



# Predict the value of the digit on the test subset
predicted2 = rf.predict(X_train)

print(f"Classification report for classifier {rf}:\n"
      f"{metrics.classification_report(y_train, predicted2)}\n")

#pred=np.array(pred)
conf=confusion_matrix(y_test, predicted)
print('Confusion Matrix')
print (conf)
confNorm = conf/conf.astype(np.float).sum(axis=1)

print('Normalized Confusion Matrix')
print(confNorm)

df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))

plt.title('Normalized Confusion Matrix \n Random Forest')
sns.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Labels", labelpad=18)
plt.ylabel("Expected Labels",labelpad=18 )
plt.show()

np.fill_diagonal(confNorm, 0)
df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))

plt.title('Absolute Confusion Matrix \n Random Forest')
sns.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Labels", labelpad=18)
plt.ylabel("Expected Labels",labelpad=18 )
plt.show()

# print(classification_report(y_test, predicted, target_names=intention))