import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

rf = RandomForestClassifier()

#import data
X=np.load("archive/TrainTestDataFlat/XSVM.npy")

y=np.load("archive/TrainTestDataFlat/ySVM.npy")

X=np.array(X)
y=np.array(y)

X_train=np.load("archive/TrainTestDataFlat/X_trainSVM.npy")
y_train=np.load("archive/TrainTestDataFlat/y_trainSVM.npy")
X_test=np.load("archive/TrainTestDataFlat/X_testSVM.npy")
y_test=np.load("archive/TrainTestDataFlat/y_testSVM.npy")
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
"""

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

pprint(rf_random.best_params_)
"""

"""
#best 
{'bootstrap': True,
 'max_depth': 100,
 'max_features': 'auto',
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 1400}
"""

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [110, 120,130,140],
    'max_features': [2, 3],
    'min_samples_leaf': [1,2],
    'min_samples_split': [1,2,3,4],
    'n_estimators': [1400, 1600, 1800, 2000, 2200]
}

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
pprint(grid_search.best_params_)
best_grid = grid_search.best_estimator_

"""
{'bootstrap': True,
 'max_depth': 110,
 'max_features': 2,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 2000}
"""