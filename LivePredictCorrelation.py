import argparse
import time
import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import tensorflow as tf
import pandas as pd
import to_text


def reformat(x):
    c1, c2, c3, c4, c5, c6, c7, c8 = [], [], [], [], [], [], [], []
    for i in range(700):
        c1.append(x[i, 0])
        c2.append(x[i, 1])
        c3.append(x[i, 2])
        c4.append(x[i, 3])
        c5.append(x[i, 4])
        c6.append(x[i, 5])
        c7.append(x[i, 6])
        c8.append(x[i, 7])

    data = {'c1': c1,
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'c5': c5,
            'c6': c6,
            'c7': c7,
            'c8': c8}

    df = pd.DataFrame(data, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'])
    correlation = df.corr()

    return correlation


class OpenBCI:
    def __init__(self):
        to_text.textbox("Trying to connect to board...\n")
        try:
            action = ["Backward", "Forward", "Left", "Right"]
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

            self.board = BoardShim(args.board_id, params)

            self.board.prepare_session()

            self.board.start_stream(45000, args.streamer_params)  # ring buffer int

            to_text.textbox("Connected!\n")

        except brainflow.board_shim.BrainFlowError:
            to_text.textbox("Connection failed.\n")

    def predict(self):
        while True:
            time.sleep(5)  # time streamed in seconds
            data = self.board.get_board_data()  # this is a numpy.ndarray

            # self.board.stop_stream()

            data = data[1:9, 1:701]  # reshaping data into same format
            data = data.transpose()
            data = reformat(data)
            data = np.array(data).reshape(-1, 8, 8)

            try:
                model = tf.keras.models.load_model("CNN Corr.h5")

                prediction = model.predict(data)

                to_text.textbox(f'The action you are thinking is: {prediction}\n')
            except OSError:
                to_text.textbox("Error, model does not exist.")
