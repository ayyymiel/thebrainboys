import numpy as np

file = np.load('ConvNet Dataset/X.npy', encoding='latin1')

file_list = np.ndarray.tolist(file)
reshapen = [elem for twod in file_list for elem in twod]
reshapen = np.array(reshapen)

reshapen = reshapen / 10000

reshapen = np.ndarray.tolist(reshapen)
the_list = [0, 1, 2, 3, 4]
row_counter, counter = 0, 0
index = 0
for i in range(len(reshapen)):  # for each row
    reshapen_row = reshapen[i]
    reshapen_row.append(the_list[index])
    row_counter += 1
    counter += 1
    if counter == 720:
        index += 1
        counter = 0

new_X_array = np.array(reshapen)
print(new_X_array.shape)
np.save('new_X_short_scaled_array', new_X_array)
print("File Saved")
