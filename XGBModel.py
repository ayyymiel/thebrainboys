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
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import joblib

action = ["Backward", "Forward", "Left", "Other", "Right"]

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

model = XGBClassifier(silent=False,
                      scale_pos_weight=1,
                      learning_rate=0.01,
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic',
                      n_estimators=1000,
                      reg_alpha = 0.3,
                      max_depth=4,
                      gamma=10)

model.fit(X_train,y_train)

#save model
joblib.dump(model, "xgbModel.joblib")

"""
#to load model 
# load, no need to initialize the loaded_rf
model = joblib.load("xgbModel.joblib")
"""
# Predict the value of the digit on the test subset
predicted = model.predict(X_test)

print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")



# Predict the value of the digit on the test subset
predicted2 = model.predict(X_train)

print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_train, predicted2)}\n")

# save the classifier
with open('SVMModel.pkl', 'wb') as fid:
    cPickle.dump(model, fid)

#pred=np.array(pred)
conf=confusion_matrix(y_test, predicted)
print('Confusion Matrix')
print (conf)
confNorm = conf/conf.astype(np.float).sum(axis=1)
print('Normalized Confusion Matrix')
print(confNorm)

df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))
plt.title('Normalized Confusion Matrix \n XGBoost')
sns.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Labels", labelpad=18)
plt.ylabel("Expected Labels",labelpad=18 )
plt.show()

np.fill_diagonal(confNorm, 0)
df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))

plt.title('Absolute Confusion Matrix \n Naive Bayes')
sns.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Labels", labelpad=18)
plt.ylabel("Expected Labels",labelpad=18 )
plt.show()