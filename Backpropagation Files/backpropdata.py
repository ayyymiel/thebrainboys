import numpy as np

file = np.load('X_train_b_2.npy', encoding='latin1')

file_list = np.ndarray.tolist(file)

reshapen = [elem for twod in file_list for elem in twod]

the_list = [0, 1, 2, 3, 4]
row_counter, counter, index = 0, 0, 0
for i in range(len(reshapen)):  # for each row
    reshapen_row = reshapen[i]
    reshapen_row.append(the_list[index])
    row_counter += 1
    counter += 1
    if counter == 90:
        index += 1
        counter = 0

new_X_array = np.array(reshapen)
print(new_X_array.shape)
np.save('new_X_short_array', new_X_array)
print("File Saved")
