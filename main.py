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
import subprocess
import datetime

epochs = 10
batch_size = 5
WEIGHTS_FILENAME = 'weights.bin'

IMG_SHAPE = (144, 256, 1)
FPS = 3
SOURCE_FPS = 24
MAX_PIXEL_VALUE = 255
VIDEO_PADDING_TIME = 1
POSITIVE_THRESHOLD = 0.5
PREPROC_STR = '_preproc'
OUTPUT_FILENAME = 'gentelmen_stuff.mp4'

backward_time_step_seconds = 10
forward_time_step_seconds = 0
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


def generate_ground_truth_from_time_frames(source_fps, preproc_fps, preproc_frames, source_marked):
    values = np.full(shape=(1, preproc_frames), fill_value=0)[0]
    preproc_duration = preproc_frames / preproc_fps

    # preproc_duration is also source_duration
    source_frames = preproc_duration * source_fps
    for time_range in source_marked:
        source_from = time_range[0] / preproc_duration * source_frames
        source_to = time_range[1] / preproc_duration * source_frames
        preproc_from = round(source_from * preproc_fps / source_fps)
        preproc_to = round(source_to * preproc_fps / source_fps)
        values[preproc_from:preproc_to + 1] = 1

    return values


def generate_ground_truth_from_frames(preproc_frames, source_marked):
    values = np.full(shape=(1, preproc_frames), fill_value=0)[0]
    for frames in source_marked:
        frame_from = frames[0]
        frame_to = frames[1]
        values[frame_from:frame_to + 1] = 1

    return values


def generate_time_frames_from_binary_vec(preproc_fps, binary_vec):
    time_frames = []
    for i in range(binary_vec.shape[0]):
        if i + 1 < binary_vec.shape[0] and binary_vec[i] < POSITIVE_THRESHOLD <= binary_vec[i + 1]:
            preproc_from = i + 1
        elif ((i + 1 < binary_vec.shape[0] and binary_vec[i] >= POSITIVE_THRESHOLD > binary_vec[i + 1])
              or (i + 1 >= binary_vec.shape[0] and binary_vec[i] >= POSITIVE_THRESHOLD)):
            preproc_to = i
            source_from = round(preproc_from / preproc_fps)
            source_to = round(preproc_to / preproc_fps)
            time_frames.append((source_from, source_to))

    return time_frames

def ffmpeg_extract_video_parts(input_filename, time_frames, output_filename):
    videos = []
    video_index = 0
    for time_frame in time_frames:
        time_from = time_frame[0] - VIDEO_PADDING_TIME
        time_from = max(time_from, 0)
        duration = time_frame[1] + VIDEO_PADDING_TIME - time_frame[0]
        video_filename = "tmp.vid." + (str(video_index) if video_index > 9 else "0" + str(video_index)) + ".mp4"
        ffmpeg_extract_video_part(input_filename, time_from, duration, output_filename)
        videos.append(video_filename)

def ffmpeg_extract_video_part(input_filename, time_from, duration, output_filename):
    subprocess.run(["ffmpeg",
                    "-ss", str(datetime.timedelta(seconds=time_from)),
                    "-i", input_filename,
                    "-t", str(duration),
                    "-vcodec", "copy",
                    "-acodec", "copy",
                    "-y", output_filename])


def main():
    # CLI parser
    parser = argparse.ArgumentParser(description='EcchiNET.')
    parser.add_argument('-i', '--input', help='Input video name', required=True)
    parser.add_argument('-p', '--predict', action="store_true", help='Predict instead of training', default=False,
                        required=False)
    parser.add_argument('-s', '--save_weights', action="store_true", help='Save weights', default=False, required=False)
    args = parser.parse_args()
    video_filename = args.input

    preproc_video_filename = video_filename.split('.mp4')[0] + PREPROC_STR + ".mp4"
    print('loading file: ', preproc_video_filename)

    container = av.open(preproc_video_filename)
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0]

    print('video stream res:', video_stream.codec_context.format.width,
          'x', video_stream.codec_context.format.height, 'frames:', video_stream.frames)

    video_data = np.asarray([np.reshape(frame.to_ndarray(format='gray'), newshape=IMG_SHAPE) for frame in
                             container.decode(video=0)]) / MAX_PIXEL_VALUE
    # video_context_data = preprocess_lstm_context(video_data)

    # ground_truth = generate_ground_truth_from_time_frames(SOURCE_FPS, FPS, video_stream.frames, [(32, 47), (93, 110), (131, 137), (143, 156)])

    ground_truth = generate_ground_truth_from_frames(video_stream.frames,
                                                     [(780, 1158), (2239, 2671), (2762, 2944), (3132, 3316),
                                                      (3427, 3538), (3544, 3643), (3649, 3749)])

    video_data = video_data[::int(SOURCE_FPS / FPS)]
    ground_truth = ground_truth[::int(SOURCE_FPS / FPS)]

    data_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True
    )
    data_generator.fit(video_data)

    model = Sequential([
        Conv2D(input_shape=(144, 256, 1), filters=16, activation='relu', kernel_size=5,
               padding='valid', strides=(2, 2), data_format='channels_last'),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format='channels_last'),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    do_predict = args.predict
    if do_predict:
        model.load_weights(WEIGHTS_FILENAME)
        predictions = model.predict(video_data)
        time_frames = generate_time_frames_from_binary_vec(FPS, predictions)
        print(time_frames)
        ffmpeg_extract_video_parts(video_filename, time_frames, OUTPUT_FILENAME)

    else:
        history = model.fit(video_data, ground_truth, batch_size=batch_size,
                            epochs=epochs, shuffle=True)
        if args.save_weights:
            model.save_weights(WEIGHTS_FILENAME)
    print('おはようございます！')


if __name__ == '__main__':
    main()
