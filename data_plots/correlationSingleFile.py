import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt

# select file to load here
# load_file = np.loadtxt('C:/Users/Amiel/PycharmProjects/thebrainboys/TrimmedData/Right/1629944775.npy', delimiter=',')
#load_file = np.loadtxt('ActionsData/Backward/1629944584.npy', delimiter=',')


X=np.load("NewConvData/X.npy")
y=np.load("NewConvData/y.npy")


X=np.array(X)
y=np.array(y)



#print(load_file)
print(X.shape)
temp=X[0].transpose()
print(pd.DataFrame(temp))

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

#correlation data
c=temp

c = reformat(c)
print(c,'\n')

sn.heatmap(c, annot=True)
plt.show()

#numpy of extra zeros needed
z=np.zeros((692,8))

#append rows zeros to add up to 700 rows
d=np.append(c,z,axis=0)
print(d,'\n')

#add correlations columns to regular data
final=np.append(temp,d,axis=1)

#if you aren't running in notebook/colab and want to view entrie dataframe, uncomment next 2 lines
#pd.set_option('display.width', 200)
#pd.set_option("display.max_rows", None, "display.max_columns", None)

print(pd.DataFrame(final),'\n')

print(X.shape,'\n')
