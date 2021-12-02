import argparse
import time
import numpy as np
import pandas as pd
import os
from collections import deque
# import cv2
import tensorflow as tf
import random
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from numpy import asarray
from numpy import savetxt
import matplotlib.pyplot as plt
from matplotlib import style
import joblib
# import keyboard
import pickle as cPickle

def start_connect():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM3')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=0)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    from sklearn.preprocessing import MinMaxScaler

    board = BoardShim(args.board_id, params)

    action = ["Backward", "Forward", "Left", "Right"]

    # with open('SVMModel.pkl', 'rb') as fid:
    #     clf = cPickle.load(fid)
    clf = joblib.load('./FFT_KNNModel.joblib')

    board.prepare_session()
    board.start_stream(45000, args.streamer_params)  # ring buffer int

    time.sleep(5)  # time streamed in seconds (+5)
    # time.sleep(10)  # time streamed in seconds

    data = board.get_board_data()  # This is a numpy.ndarray

    name=f"{int(time.time())}.npy"
    DataFilter.write_file(data, name, 'w')  # use 'a' for append mode
    # Don't use latest brainflow version, it will cause grief: use 3.9.2

    data = data[1:9, 100:800]
    data = np.array(data).reshape(-1, 8, 700)
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    data = scaler.fit_transform(data)
    prediction = clf.predict(data)
    print("The action you are thinking is: ", prediction, " from data")


    x = np.loadtxt(name, delimiter=',')
    x = np.array(x)

    x = np.transpose(x)
    x=x[1:9, 100:800]
    x = np.array(x).reshape(-1, 8, 700)
    n_samples = len(x)
    x = x.reshape((n_samples, -1))
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    x = scaler.fit_transform(x)
    prediction = clf.predict(x)

    print("The action you are thinking is: ", prediction, "from file")


start_connect()