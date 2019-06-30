import argparse
import numpy as np
#import pandas as pd
#from keras.layers import LSTM, Dense
#from keras.models import Sequential
#from sklearn.preprocessing import LabelEncoder
import time
import av
from matplotlib import pyplot as plt

epochs = 15
batch_size = 10

LSTM_nodes = 64
backward_time_step = 3
forward_time_step = 3
dropout = 0.2

def main():
    # CLI parser
    parser = argparse.ArgumentParser(description='EcchiNET.')
    parser.add_argument('-i', '--input', help='Input video name', required=True)
    args = parser.parse_args()
    video_filename = args.input

    print('loading file: ', video_filename)

    container = av.open(video_filename)
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0]

    print('video stream res:', video_stream.codec_context.format.width,
          'x', video_stream.codec_context.format.height, 'frames:', video_stream.frames)

    video_data = np.asarray([frame.to_rgb().to_ndarray() for frame in container.decode(video=0)])
    plt.imshow(video_data[int(video_stream.frames / 2), :, :])
    plt.show()


    print('おはようございます！')

if __name__ == '__main__':
    main()
