import argparse
import numpy as np
# import pandas as pd
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, TimeDistributed, Flatten
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
import time
import av
from matplotlib import pyplot as plt
import subprocess
from utils import utils

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

    ground_truth = utils.generate_ground_truth_from_frames(video_stream.frames,
                                                     [(780, 1158), (2239, 2671), (2762, 2944), (3132, 3316),
                                                      (3427, 3538), (3544, 3643), (3649, 3749)])

    video_data = video_data[::int(SOURCE_FPS / FPS)]
    ground_truth = ground_truth[::int(SOURCE_FPS / FPS)]

    container.seek(offset=0)
    audio_data = np.asarray([frame.to_ndarray() for frame in container.decode(audio=0)])
    audio_data_newshape = (video_data.shape[0], int(audio_data.size / (video_data.shape[0] * 2 * audio_data.shape[2])), 2, audio_data.shape[2])
    audio_data = utils.reshape_prune_extra(audio_data, dst_shape=audio_data_newshape)

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
        time_frames = utils.generate_time_frames_from_binary_vec(FPS, predictions, POSITIVE_THRESHOLD)
        print(time_frames)
        videos = utils.ffmpeg_extract_video_parts(video_filename, time_frames, OUTPUT_FILENAME, VIDEO_PADDING_TIME)
        for video in videos:
            subprocess.run(["rm", video])

    else:
        history = model.fit(video_data, ground_truth, batch_size=batch_size,
                            epochs=epochs, shuffle=True)
        if args.save_weights:
            model.save_weights(WEIGHTS_FILENAME)
    print('おはようございます！')


if __name__ == '__main__':
    main()
