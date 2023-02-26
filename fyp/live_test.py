from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt
from dataset_tools import check_std_deviation, BOARD_SAMPLING_RATE, preprocess_raw_eeg
import keras
import numpy as np
import argparse
import time

model = keras.models.load_model('./models_2_71/fold_n5/5_model.h5')

if __name__ == '__main__':

    NUM_CHANNELS = 8
    NUM_TIMESTAMP_PER_SAMPLE = 250

    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port',
                        required=False, default='COM7')

    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(0, params)
    board.prepare_session()

    last_act = None

    for i in range(50):
        input("Press enter to acquire a new action")
        print("Think in 2")
        time.sleep(1.5)
        print("Think in 1")
        time.sleep(1.5)
        print("Think NOW!!")
        time.sleep(1.5)  

        board.start_stream() 
        time.sleep(1.5 * (NUM_TIMESTAMP_PER_SAMPLE / BOARD_SAMPLING_RATE))
        data = board.get_current_board_data(NUM_TIMESTAMP_PER_SAMPLE)
        board.stop_stream()

        sample = []
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        for channel in eeg_channels:
            sample.append(data[channel])

        print(np.array(sample).shape)
        
        if np.array(sample).shape == (NUM_CHANNELS, NUM_TIMESTAMP_PER_SAMPLE) and check_std_deviation(np.array(sample)):
            sample = np.array(sample)
            
            data_X, fft_data_X = preprocess_raw_eeg(sample.reshape((1, 8, 250)), lowcut=8, highcut=45, coi3order=0)
            nn_input = data_X.reshape((1, 8, 250, 1)) 
            result = model.predict(nn_input)
            result = np.argmax(result)
            if result == 0:
                print("feet")
            elif result == 1:
                print("Hands")
            elif result == 2:
                print("None")        

        plt.show()
       
    board.release_session()
