import argparse
import time
import numpy as np
import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# from mne.viz.topomap import _prepare_topo_plot, plot_topomap
from mne.viz.topomap import plot_topomap
import mne
from mne.channels import read_layout

from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    BoardShim.enable_dev_board_logger ()
    
    # same as brainflow example except they used synth. board
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
    # ------------------------------------------------------

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

    # params = BrainFlowInputParams ()
    # board = BoardShim (BoardIds.CYTON_BOARD.value, params)
    board.prepare_session ()
    board.start_stream ()
    time.sleep (30)                         # <-- this long so that MNE doesn't complain about lack of data.
    data = board.get_board_data ()
    board.stop_stream ()
    board.release_session ()

    eeg_channels = BoardShim.get_eeg_channels (BoardIds.CYTON_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000           # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    ch_names = ['T7', 'CP5', 'FC5', 'C3', 'C4', 'FC6', 'CP6', 'T8']
    sfreq = BoardShim.get_sampling_rate (BoardIds.CYTON_BOARD.value)
    info = mne.create_info (ch_names = ch_names, sfreq = sfreq, ch_types = ch_types)
    raw = mne.io.RawArray (eeg_data, info)

    # do something to data (preprocessing)
    raw.notch_filter(np.arange(60, 125, 60))  

    # its time to plot something!
    raw.plot_psd (average = True)
    plt.savefig ('psd.png')


if __name__ == '__main__':
    main ()