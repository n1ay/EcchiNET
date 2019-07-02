import argparse
import numpy as np
# import pandas as pd
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, TimeDistributed, Flatten
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
# from sklearn.preprocessing import LabelEncoder
import time
import av
from matplotlib import pyplot as plt

epochs = 10
batch_size = 1

FPS = 4
backward_time_step_seconds = 1
forward_time_step_seconds = 1
backward_time_step = backward_time_step_seconds * FPS
forward_time_step = forward_time_step_seconds * FPS
frames_per_sample = backward_time_step + 1 + forward_time_step


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


def generate_ground_truth(source_fps, preproc_fps, preproc_frames, source_marked):
    values = np.full(shape=(1, preproc_frames), fill_value=0)[0]
    preproc_duration = preproc_frames / preproc_fps

    # preproc_duration is also source_duration
    source_frames = preproc_duration * source_fps
    for time_range in source_marked:
        source_from = time_range[0] / preproc_duration * source_frames
        source_to = time_range[1] / preproc_duration * source_frames
        preproc_from = round(source_from * preproc_fps / source_fps)
        preproc_to = round(source_to * preproc_fps / source_fps)
        values[preproc_from:preproc_to] = 1

    return values


def generate_time_frames_from_binary_vec(preproc_fps, binary_vec):
    time_frames = []
    for i in range(binary_vec.shape[0]):
        if i + 1 < binary_vec.shape[0] and binary_vec[i] == 0 and binary_vec[i + 1] == 1:
            preproc_from = i + 1
        elif ((i + 1 < binary_vec.shape[0] and binary_vec[i] == 1 and binary_vec[i + 1] == 0)
              or (i + 1 >= binary_vec.shape[0] and binary_vec[i] == 1)):
            preproc_to = i
            source_from = round(preproc_from / preproc_fps)
            source_to = round(preproc_to / preproc_fps)
            time_frames.append((source_from, source_to))

    return time_frames


def main():
    # CLI parser
    parser = argparse.ArgumentParser(description='EcchiNET.')
    parser.add_argument('-i', '--input', help='Input video name', required=True)
    parser.add_argument('-p', '--predict', action="store_true", help='Predict instead of training', default=False,
                        required=False)
    parser.add_argument('-s', '--save_weights', action="store_true", help='Save weights', default=False, required=False)
    args = parser.parse_args()
    video_filename = args.input

    print('loading file: ', video_filename)

    container = av.open(video_filename)
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0]

    print('video stream res:', video_stream.codec_context.format.width,
          'x', video_stream.codec_context.format.height, 'frames:', video_stream.frames)

    video_data = np.asarray([frame.to_rgb().to_ndarray() for frame in container.decode(video=0)])
    # video_context_data = preprocess_lstm_context(video_data)

    ground_truth = generate_ground_truth(24, FPS, video_stream.frames, [(32, 47), (93, 110), (131, 137), (143, 156)])

    data_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True
    )

    model = Sequential([
        Conv2D(input_shape=(180, 240, 3), filters=32, activation='relu', kernel_size=5,
               padding='valid', strides=(1, 1), data_format='channels_last'),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format='channels_last'),
        Flatten(),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    do_predict = args.predict
    if do_predict:
        model.load_weights('weights.bin')
        predictions = model.predict(video_data)
        time_frames = generate_time_frames_from_binary_vec(FPS, predictions)
        print(time_frames)

    else:
        history = model.fit(video_data, ground_truth, batch_size=batch_size, epochs=epochs, shuffle=True)
        if args.save_weights:
            model.save_weights('weights.bin')
    print('おはようございます！')


if __name__ == '__main__':
    main()
