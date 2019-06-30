import argparse
import numpy as np
# import pandas as pd
# from keras.layers import LSTM, Dense
# from keras.models import Sequential
# from sklearn.preprocessing import LabelEncoder
import time
import av
from matplotlib import pyplot as plt

epochs = 15
batch_size = 10

LSTM_nodes = 64
FPS = 1
backward_time_step_seconds = 1
forward_time_step_seconds = 0
backward_time_step = backward_time_step_seconds * FPS
forward_time_step = forward_time_step_seconds * FPS
dropout = 0.2


def preprocess_lstm_context(data):
    context_data = []
    for i in range(data.shape[0]):
        frame_context = []
        for j in range(backward_time_step, 0, -1):
            if i < j:
                frame_context.append(data[0])
            else:
                frame_context.append(data[i - j])
        frame_context.append(data[i])
        for j in range(1, forward_time_step + 1):
            if i + j > data.shape[0] - 1:
                frame_context.append(data[-1])
            else:
                frame_context.append(data[i + j])
        context_data.append(np.asarray(frame_context))
    return np.asarray(context_data)


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
    video_context_data = preprocess_lstm_context(video_data)

    print('おはようございます！')


if __name__ == '__main__':
    main()
