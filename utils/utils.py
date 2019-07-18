import numpy as np
import datetime
import subprocess
from functools import reduce


def preprocess_lstm_context(data: np.array, backward_time_step: int, forward_time_step: int):
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


def generate_ground_truth_from_frames(preproc_frames_num, true_marked_frame_ranges):
    values = np.full(shape=(1, preproc_frames_num), fill_value=0)[0]
    for frames in true_marked_frame_ranges:
        frame_from = frames[0]
        frame_to = frames[1]
        values[frame_from:frame_to + 1] = 1

    return values


def generate_time_frames_from_binary_vec(preproc_fps, binary_vec, positive_threshold):
    time_frames = []
    preproc_from = 0
    for i in range(binary_vec.shape[0]):
        if i + 1 < binary_vec.shape[0] and binary_vec[i] < positive_threshold <= binary_vec[i + 1]:
            preproc_from = i + 1
        elif ((i + 1 < binary_vec.shape[0] and binary_vec[i] >= positive_threshold > binary_vec[i + 1])
              or (i + 1 >= binary_vec.shape[0] and binary_vec[i] >= positive_threshold)):
            preproc_to = i
            source_from = round(preproc_from / preproc_fps)
            source_to = round(preproc_to / preproc_fps)
            time_frames.append((source_from, source_to))

    return time_frames


def ffmpeg_extract_video_parts(input_filename, time_frames, output_filename, video_padding_time):
    videos = []
    video_index = 0
    for time_frame in time_frames:
        time_from = time_frame[0] - video_padding_time
        time_from = max(time_from, 0)
        duration = time_frame[1] + video_padding_time - time_frame[0]
        video_filename = "tmp.vid." + (str(video_index) if video_index > 9 else "0" + str(video_index)) + ".mp4"
        ffmpeg_extract_video_part(input_filename, time_from, duration, video_filename)
        videos.append(video_filename)
        video_index += 1

    subprocess.run(["./ffmpeg_concat.sh", output_filename])
    return videos


def ffmpeg_extract_video_part(input_filename, time_from, duration, output_filename):
    subprocess.run(["ffmpeg",
                    "-ss", str(datetime.timedelta(seconds=time_from)),
                    "-i", input_filename,
                    "-t", str(duration),
                    "-vcodec", "copy",
                    "-acodec", "copy",
                    "-y", output_filename])


def reshape_prune_extra(array, dst_shape):
    array_1d = np.reshape(array, newshape=array.size)
    dst_elements = reduce(lambda x, y: x * y, dst_shape)
    array_1d = array_1d[0: dst_elements]
    return np.reshape(array_1d, dst_shape)
