import numpy as np

get_data1 = np.load("JumbleConvNetData\X (1).npy")
get_data2 = np.load("JumbleConvNetData\X_test.npy")
get_data3 = np.load("JumbleConvNetData\X_train.npy")
get_data4 = np.load("JumbleConvNetData\y_test.npy")
get_data5 = np.load("JumbleConvNetData\y_train.npy")
get_data6 = np.load("JumbleConvNetData\y.npy")

print(np.shape(get_data1))
print(np.shape(get_data2))
print(np.shape(get_data3))
print(np.shape(get_data4))
print(np.shape(get_data5))
print(np.shape(get_data6))