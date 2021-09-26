import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt

# select file to load here
load_file = np.loadtxt('C:/Users/Amiel/PycharmProjects/thebrainboys/TrimmedData/Right/1629944775.npy', delimiter=',')


def reformat():
    c1, c2, c3, c4, c5, c6, c7, c8 = [], [], [], [], [], [], [], []
    for i in range(len(load_file)):
        c1.append(load_file[i, 0])
        c2.append(load_file[i, 1])
        c3.append(load_file[i, 2])
        c4.append(load_file[i, 3])
        c5.append(load_file[i, 4])
        c6.append(load_file[i, 5])
        c7.append(load_file[i, 6])
        c8.append(load_file[i, 7])

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


c = reformat()

sn.heatmap(c, annot=True)
plt.show()
