import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

dataset = np.load('Backpropagation Files/new_X_array.npy')
dataset = np.delete(dataset, 5600, axis=1)
dataset = dataset / 1000
dataset = np.ndarray.tolist(dataset)
removed_list = []

for i in range(len(dataset)):
    removed = dataset[i].pop(5598)
    removed_list.append(removed)

dataset = np.array(dataset)
avg_list = np.mean(dataset, axis=1)
new_list = np.reshape(avg_list, (-1,1))
new_list = np.ndarray.tolist(new_list)

type_list = [1, 2, 3, 4, 5]
counter = 0
index = 0
for i in range(len(new_list)):
    new_list[i].append(removed_list[i])
    new_list[i].append(type_list[index])
    counter += 1
    if counter == 66:
        counter = 0
        index += 1

new_df = pd.DataFrame(new_list)

dbscan_opt = DBSCAN(eps=2.2, min_samples=5)
dbscan_opt.fit(new_df[[0, 1]])
new_df['DBSCAN_opt_labels'] = dbscan_opt.labels_

colors = ['purple', 'red', 'blue', 'green', 'cyan', 'pink']
plt.figure(figsize=(10,10))
plt.scatter(new_df[0],new_df[1],c=new_df['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering',fontsize=20)
for i, txt in enumerate(new_df[2]):
    plt.annotate(txt, (new_df[0][i],new_df[1][i]))
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()
