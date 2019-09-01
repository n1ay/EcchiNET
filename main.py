import argparse
import numpy as np
# import pandas as pd

import time
import av
from matplotlib import pyplot as plt
import subprocess
from utils import utils
from model import *

video_epochs = 5
audio_epochs = 10
merging_epochs = 15
batch_size = 5
WEIGHTS_VIDEO_FILENAME = 'weights_video.bin'
WEIGHTS_AUDIO_FILENAME = 'weights_audio.bin'
WEIGHTS_MERGING_FILENAME = 'weights_merging.bin'
DATA_DESCRIPTION_FILENAME = 'data.json'

FPS = 3
SOURCE_FPS = 24
MAX_PIXEL_VALUE = 255
VIDEO_PADDING_TIME = 1
POSITIVE_THRESHOLD = 0.8
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
    parser.add_argument('-i', '--input', help='Input video name(s)', required=True, nargs='+')
    parser.add_argument('-p', '--predict', action="store_true", help='Predict instead of training', default=False,
                        required=False)
    parser.add_argument('-s', '--save_weights', action="store_true", help='Save weights', default=False, required=False)
    args = parser.parse_args()

    # prepare data
    data_description_map = {}
    data_description = utils.load_parse_data_description(DATA_DESCRIPTION_FILENAME)
    for description_entry in data_description:
        data_description_map[description_entry['path']] = description_entry['ground_truth']
    video_data_full = []
    video_data_ground_truth_full = []
    audio_data_full = []
    audio_data_ground_truth_full = []

    for video_filename in args.input:

        preproc_video_filename = video_filename.split('.mp4')[0] + PREPROC_STR + ".mp4"
        print('loading file: ', preproc_video_filename)

        container = av.open(preproc_video_filename)
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]

        print('video stream res:', video_stream.codec_context.format.width,
              'x', video_stream.codec_context.format.height, 'frames:', video_stream.frames)

        video_data = np.asarray([np.reshape(frame.to_ndarray(format='gray'), newshape=IMG_SHAPE) for frame in
                                 container.decode(video=0)]) / MAX_PIXEL_VALUE
        # video_context_data = utils.preprocess_lstm_context(video_data)

        video_data_ground_truth = utils.generate_ground_truth_from_frames(video_stream.frames, data_description_map[video_filename])
        video_data = video_data[::int(SOURCE_FPS / FPS)]
        video_data_ground_truth = video_data_ground_truth[::int(SOURCE_FPS / FPS)]

        container.seek(offset=0)
        audio_data = np.asarray([frame.to_ndarray() for frame in container.decode(audio=0)])
        audio_data_ground_truth = np.asarray([video_data_ground_truth[int(idx * video_data.shape[0] / audio_data.shape[0])] for idx in range(audio_data.shape[0])])

        print('audio_data: ', audio_data[0].shape, '\n\n\nvideo_data: ', video_data[0].shape)

        video_data_full.append(video_data)
        video_data_ground_truth_full.append(video_data_ground_truth)
        audio_data_full.append(audio_data)
        audio_data_ground_truth_full.append(audio_data_ground_truth)

    # data_generator = build_data_generator()
    # data_generator.fit(video_data)

    model_video = build_model_video()
    model_audio = build_model_audio()
    model_merging = build_model_merging()

    do_predict = args.predict
    if do_predict:
        model_video.load_weights(WEIGHTS_VIDEO_FILENAME)
        model_audio.load_weights(WEIGHTS_AUDIO_FILENAME)
        model_merging.load_weights(WEIGHTS_MERGING_FILENAME)
        predictions_video = model_video.predict(video_data)
        predictions_audio = model_audio.predict(audio_data)
        predictions_audio_processed = utils.reshape_average_prune_extra(predictions_audio, predictions_video.shape[0])
        predictions_video_audio = np.column_stack([predictions_video, predictions_audio_processed])
        predictions = model_merging.predict(predictions_video_audio)
        time_frames = utils.generate_time_frames_from_binary_vec(FPS, VIDEO_PADDING_TIME, predictions,
                                                                 POSITIVE_THRESHOLD)
        print(time_frames)
        videos = utils.ffmpeg_extract_video_parts(video_filename, time_frames, OUTPUT_FILENAME)
        for video in videos:
            subprocess.run(["rm", video])

    else:
        history_model_video = model_video.fit(video_data, video_data_ground_truth, batch_size=batch_size,
                                              epochs=video_epochs, shuffle=True)
        history_model_audio = model_audio.fit(audio_data, audio_data_ground_truth, batch_size=batch_size,
                                              epochs=audio_epochs, shuffle=True)
        predictions_video = model_video.predict(video_data)
        predictions_audio = model_audio.predict(audio_data)
        predictions_audio_processed = utils.reshape_average_prune_extra(predictions_audio, predictions_video.shape[0])
        predictions_video_audio = np.column_stack([predictions_video, predictions_audio_processed])

        history_model_merging = model_merging.fit(predictions_video_audio, video_data_ground_truth,
                                                  batch_size=batch_size, epochs=merging_epochs, shuffle=True)
        if args.save_weights:
            model_video.save_weights(WEIGHTS_VIDEO_FILENAME)
            model_audio.save_weights(WEIGHTS_AUDIO_FILENAME)
            model_merging.save_weights(WEIGHTS_MERGING_FILENAME)
    print('おはようございます！')


if __name__ == '__main__':
    main()
