import argparse
import time
import numpy as np
# import serial
from pathlib import Path

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from numpy.core.numeric import NaN
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import pickle
import joblib

action = ["Backward", "Forward", "Left", "Other", "Right"]
# ser = serial.Serial("COM4", 9600, timeout = 1) #Change your port name COM... and your baudrate
BoardShim.enable_dev_board_logger()

parser = argparse.ArgumentParser()
# use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                    default=0)
parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                    default=0)
parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
# parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='/dev/cu.usbserial-DM0258P6')   # <- mac
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM3')  # <- windows
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

board.prepare_session()

board.start_stream(45000, args.streamer_params)  # ring buffer int

# model = tf.keras.models.load_model("C:/Users/gozon/PycharmProjects/thebrainboys/FFT_KNNModel.joblib")
model = joblib.load("C:/Users/gozon/PycharmProjects/thebrainboys/FFT SVMModel.joblib")

def main():
    time.sleep(5)  # time streamed in seconds

    data = board.get_board_data()  # This is a numpy.ndarray

    # board.stop_stream()

    data = data[1:9, 1:701]  # reshaping data into same format
    data = np.array(data).reshape(-1, 8, 700)
    data=np.transpose(data)
    data=np.fft.fft(data)
    data=np.abs(data)
    data = np.array(data).reshape(-1, 8, 700)
    n_samples = len(data)
    data = data.reshape((n_samples, -1))  # flatten array essentially turning it into a 1d array (8,700) --> (5600,)
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    data = scaler.fit_transform(data)

    prediction = model.predict(data)
    

    print("The action you are thinking is: ", prediction)
    print("The action you are thinking is: ", action[np.argmax(prediction)])



if __name__ == "__main__":
    # while(True==True):
    main()