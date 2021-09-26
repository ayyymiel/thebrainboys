import numpy as np

file = np.load('X_train_b.npy')
print(len(file[0,0]))


# reshapen = np.reshape(file, (330, 8*700))
# reshapen = np.ndarray.tolist(reshapen)
#
# the_list = [0, 1, 2, 3, 4]
# row_counter, counter, index = 0, 0, 0
# for i in range(len(reshapen)):  # for each row
#     reshapen_row = reshapen[i]
#     reshapen_row.append(the_list[index])
#     row_counter += 1
#     counter += 1
#     if counter == 66:
#         index += 1
#         counter = 0
#
# new_X_array = np.array(reshapen)
# print(new_X_array.shape)
# np.save('new_X_array', new_X_array)
# print("File Saved")