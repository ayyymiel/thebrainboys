# Capstone GUI
# === Cannot run without prior connections ===
from tkinter import *

import numpy
import serial
import sys
import threading
import argparse
import time
import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import tensorflow as tf
import pandas as pd
import to_text


class ArduinoBot:
    def __init__(self):
        self.ser = serial.Serial("COM4", 9600, timeout=1)

    def move_b(self):
        self.ser.write(b'b')

    def move_f(self):
        self.ser.write(b'f')

    def move_l(self):
        self.ser.write(b'l')

    def move_r(self):
        self.ser.write(b'r')


arduino = ArduinoBot()


def start_gui():

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
                parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection',
                                    required=False,
                                    default=0)
                parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
                parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum',
                                    required=False,
                                    default=0)
                parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
                parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM3')
                parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
                parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
                parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
                parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
                parser.add_argument('--board-id', type=int,
                                    help='board id, check docs to get a list of supported boards',
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

                to_text.textbox("Connected to helmet!\n")

            except brainflow.board_shim.BrainFlowError:
                to_text.textbox("OpenBCI Connection failed.\n")

        def predict(self):
            while True:
                time.sleep(7)  # time streamed in seconds
                data = self.board.get_board_data()  # this is a numpy.ndarray

                data = data[1:9, 1:701]  # reshaping data into same format
                data = data.transpose()
                data = reformat(data)
                data = np.array(data).reshape(-1, 8, 8)

                try:
                    model = tf.keras.models.load_model("CNN Corr.h5")
                    prediction = model.predict(data)
                    to_text.textbox(f'The action you are thinking is: {prediction}\n')
                    high_index = numpy.argmax(prediction)
                    if high_index == 0:
                        arduino.move_b()
                        t.insert(END, 'Moving backward\n')
                    elif high_index == 1:
                        arduino.move_f()
                        t.insert(END, 'Moving forward\n')
                    elif high_index == 2:
                        arduino.move_l()
                        t.insert(END, 'Moving left\n')
                    elif high_index == 3:
                        arduino.move_r()
                        t.insert(END, 'Moving right\n')
                except OSError:
                    to_text.textbox("Error, model does not exist.")

        def stop(self):
            self.board.stop_stream()

    bci = OpenBCI()
    t1 = threading.Thread(target=bci.predict)

    root = Tk()
    root.geometry('760x500')
    f1 = Frame(root)
    f1.grid(row=0, column=0)
    l = Label(f1, text="The Brain Boys", bg="red")
    l.grid(row=0, column=0)

    # Connect to Cyton
    b = Button(f1, text="Predict", command=t1.start)
    b.grid(row=1, column=0, padx=30, pady=10)
    b = Button(f1, text="Stop", command=bci.stop)
    b.grid(row=1, column=1, padx=30, pady=10)
    f2 = Frame(root)

    # Connect with Arduino Bot
    f2.grid(row=2, column=0, pady=20)
    l = Label(f2, text="Bot Control")
    l.grid(row=0, column=0, pady=5)
    b = Button(f2, text="Connect", width=10, command=arduino)
    b.grid(row=1, column=0)
    b = Button(f2, text="Left", width=10, command=arduino.move_l)
    b.grid(row=2, column=0)
    b = Button(f2, text="Right", width=10, command=arduino.move_r)
    b.grid(row=3, column=0)
    b = Button(f2, text="Backward", width=10, command=arduino.move_b)
    b.grid(row=4, column=0)
    b = Button(f2, text="Forward", width=10, command=arduino.move_f)
    b.grid(row=5, column=0)
    f3 = Frame(root)
    f3.grid(row=1, rowspan=5, column=2)
    t = Text(f3, height=15, width=65)
    t.grid(row=0, column=3)

    def redirector(input_str):
        t.insert(END, input_str)

    sys.stdout.write = redirector

    root.mainloop()



start_gui()
