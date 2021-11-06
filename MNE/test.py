import numpy as np
import os
import random
import pickle
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import pandas as pd
import mne
from mne.channels import read_layout
import matplotlib.pyplot as plt

def joinNumpy():
    file1 = 'TrimmedData/Backward/1629944618.npy'
    file2 = 'TrimmedData/Backward/1629944601.npy'
    file3 = 'TrimmedData/Backward/1629944653.npy'

    x1=np.loadtxt(file1,delimiter=',')
    x2=np.loadtxt(file2,delimiter=',')
    x3=np.loadtxt(file3,delimiter=',')

    return np.concatenate((x1, x2, x3))

def main(): 
    file = 'TrimmedData/Backward/1629944618.npy'

    # filename=os.listdir(dir)
    x=np.loadtxt(file,delimiter=',')
    # x = joinNumpy()
    x=np.array(x)
    print('x ', type(x))
    print(x, '\n')

    x=np.transpose(x)
    print('Transposed')
    print(x, '\n')

    x = x/1000000
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    ch_names = ['T7', 'CP5', 'FC5', 'C3', 'C4', 'FC6', 'CP6', 'T8']

    sfreq = 250

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(x, info)

    # do something to data (preprocessing)
    final=raw.notch_filter(
        np.arange(60, 125, 60), 
        filter_length=690, trans_bandwidth=2.4
        )
    print('Final ', type(final))
    print(final, '\n')

    x_final=final[:][0]
    print(x_final)
    print(type(x_final), '\n')

    diff=np.where(x!=x_final)
    print('Diff')
    print(diff)
    final.plot_psd(average=True)
    plt.show()

if __name__ == "__main__":
    main()