import argparse
import time
import numpy as np
import pandas as pd
import os
from collections import deque
import cv2
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

"""
This code was used to create the directories for the left, right, none actions 
"""
def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='/dev/cu.usbserial-DM0258P6')   # <- mac
    # parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM6') # <- windows
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

    board = BoardShim(args.board_id, params)

    ACTION = 'other'  # THIS IS THE ACTION YOU'RE THINKING

    board.prepare_session()

    board.start_stream(45000, args.streamer_params)  # ring buffer int
    time.sleep(1)  # time streamed in seconds (+5)


    data = board.get_board_data() #This is a numpy.ndarray

    board.stop_stream()

    print()
    print(board.get_sampling_rate(args.board_id)) # <--- inconsistency in rows of data.

    # savetxt('raw_data.csv', data, delimiter=' ')
    #keeps the 8 channels of data for 700 rows which is about 3 seconds of data (5 seconds gives about 1130-1185 rows)
    # keyData = data[:, 1:701]

    # savetxt('key_data.csv', keyData, delimiter=' ')

    # Check data
    # check_data(keyData)

    # dataT=data.T
    # savetxt('data_T.csv', dataT, delimiter=' ')

    # DataFilter.write_file(data, 'test3.csv', 'w')  # use 'a' for append mode
    # restored_data = DataFilter.read_file('test3.csv')
    # restored_df = pd.DataFrame(np.transpose(restored_data))


    DataFilter.write_file(data, f"{int(time.time())}.npy", 'w')  # use 'a' for append mode
    # DataFilter.write_file(keyData, f"{int(time.time())}.npy", 'w')  # use 'a' for append mode
    # Don't use latest brainflow version, it will cause grief: use 3.9.2


def check_data(incoming_data):
    if (incoming_data.shape != (24, 700)):
        print("Sample too small", incoming_data.shape)


if __name__ == "__main__":
    main()
