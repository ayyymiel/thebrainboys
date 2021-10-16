from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd
import seaborn as sns

action = ["Backward", "Forward", "Left", "Other", "Right"]
#bring in data

# X=np.load("TrainTestDataFlat/NaiveBayesSets/X.npy")
# y=np.load("TrainTestDataFlat/NaiveBayesSets/y.npy")
# X_train=np.load("TrainTestDataFlat/NaiveBayesSets/X_train.npy")
# y_train=np.load("TrainTestDataFlat/NaiveBayesSets/y_train.npy")
# X_test=np.load("TrainTestDataFlat/NaiveBayesSets/X_test.npy")
# y_test=np.load("TrainTestDataFlat/NaiveBayesSets/y_test.npy")


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
print(X_train[0].shape)
gnb = GaussianNB()
#gnb = MultinomialNB() # not great
#gnb = BernoulliNB() #trash

y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# Predict the value of the digit on the test subset
predicted = gnb.predict(X_test)


print(f"Classification report for classifier {gnb}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")



# Predict the value of the digit on the test subset
predicted2 = gnb.predict(X_train)

print(f"Classification report for classifier {gnb}:\n"
      f"{metrics.classification_report(y_train, predicted2)}\n")


#pred=np.array(pred)
conf=confusion_matrix(y_test, predicted)
print('Confusion Matrix')
print (conf)
confNorm = conf/conf.astype(np.float).sum(axis=1)
print('Normalized Confusion Matrix \n Naive Bayes')
print(confNorm)

df_cm = pd.DataFrame(confNorm, index=action, columns=action)
plt.figure(figsize=(8,8))
plt.title('Normalized Confusion Matrix \n Naive Bayes')
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