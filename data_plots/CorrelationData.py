import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def reformat(x):
    c1, c2, c3, c4, c5, c6, c7, c8 = [], [], [], [], [], [], [], []
    for i in range(700):
        c1.append(x[i, 0])
        c2.append(x[i, 1])
        c3.append(x[i, 2])
        c4.append(x[i, 3])
        c5.append(x[i, 4])
        c6.append(x[i, 5])
        c7.append(x[i, 6])
        c8.append(x[i, 7])

    data = {'c1': c1,
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'c5': c5,
            'c6': c6,
            'c7': c7,
            'c8': c8}

    df = pd.DataFrame(data, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'])
    correlation = df.corr()

    return correlation

X=np.load("NewConvData/X.npy")
y=np.load("NewConvData/y.npy")

X=np.array(X)
y=np.array(y)

#length of loop (400)
xlen=X.shape[0]

newX = []
corr = []
for i in range(xlen):
    temp=X[i].transpose()
    c=temp
    c = reformat(c)
    c=c.fillna(0)
    corr.append(c)
    z = np.zeros((692, 8))
    d = np.append(c, z, axis=0)
    final = np.append(temp, d, axis=1)
    newX.append(final)

print(pd.DataFrame(newX[0]))

newX=np.array(newX).reshape(-1, 16,700)

print(newX.shape)
#print(corr)
corr=np.array(corr).reshape(-1,8,8)
print(corr.shape)


np.save("CorrDataForModels/corr.npy", corr)
np.save("CorrDataForModels/corrX.npy", newX)


n_samples = len(corr)
corr = corr.reshape((n_samples, -1)) #flatten array essentially turning it into a 1d array (8,700) --> (5600,)
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
corr = scaler.fit_transform(corr)
np.save("CorrDataForModels/FlatCorr.npy", corr)

"""

np.save("CorrData/corr_X.npy", corr)

n_samples = len(corr)
corr = corr.reshape((n_samples, -1)) #flatten array essentially turning it into a 1d array (8,700) --> (5600,)
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
corr = scaler.fit_transform(corr)
np.save("CorrData/Flatcorr_x.npy", corr)
"""