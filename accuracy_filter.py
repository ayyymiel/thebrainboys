import pandas as pd

get_accuracies = pd.read_csv('accuracy_test_2.csv')
low_end = []
for i in range(len(get_accuracies)):
    if get_accuracies.iloc[i, 0] < 0.4:
        low_end.append(i)
print(low_end)

for row in low_end:
    get_accuracies.drop(row, inplace=True)
    print(f'Deleted: {row}')
    get_accuracies.to_csv('new_acc_2.csv')
        # new = get_accuracies.drop(i, axis=0, inplace=False)
        # print(f'Deleted row {i}')
        # print(new)