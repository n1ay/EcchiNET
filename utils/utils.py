import numpy as np
import datetime
import subprocess
from functools import reduce
import json
import math


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


def generate_time_frames_from_binary_vec(preproc_fps, video_padding_time, binary_vec, positive_threshold):
    time_frames = []
    preproc_from = 0
    last_time_frame = None
    for i in range(binary_vec.shape[0]):
        if i + 1 < binary_vec.shape[0] and binary_vec[i] < positive_threshold <= binary_vec[i + 1]:
            preproc_from = i + 1
        elif ((i + 1 < binary_vec.shape[0] and binary_vec[i] >= positive_threshold > binary_vec[i + 1])
              or (i + 1 >= binary_vec.shape[0] and binary_vec[i] >= positive_threshold)):
            preproc_to = i

            source_from = round(preproc_from / preproc_fps) - video_padding_time
            source_from = max(0, source_from)
            source_to = round(preproc_to / preproc_fps) + video_padding_time
            source_to = min(int((binary_vec.shape[0] - 1) / preproc_fps), source_to)

            if last_time_frame is not None and last_time_frame[0] <= source_from <= last_time_frame[1]:
                last_time_frame[1] = max(last_time_frame[1], source_to)
            else:
                time_frame = [source_from, source_to]
                time_frames.append(time_frame)
                last_time_frame = time_frame

    return time_frames


def ffmpeg_extract_video_parts(input_filename, time_frames, output_filename):
    videos = []
    video_index = 0
    for time_frame in time_frames:
        time_from = time_frame[0]
        time_from = max(time_from, 0)
        duration = time_frame[1] - time_frame[0]
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


def reshape_average_prune_extra(vector, dst_length):
    length = vector.shape[0]
    if dst_length > length:
        return vector

    if dst_length <= 0:
        return []
    avg_group_elements = length / dst_length
    result = []
    index = 0
    accumulated_group_elements = avg_group_elements
    while len(result) < dst_length:
        current_group_elements = int(accumulated_group_elements)
        current_group = vector[index:index + current_group_elements]
        result.append(np.mean(current_group))
        index += current_group_elements
        accumulated_group_elements = accumulated_group_elements - current_group_elements + avg_group_elements

    return np.asarray(result)


def load_parse_data_description(data_desc_file):
    with open(data_desc_file, 'r') as f:
        data_desc = json.load(f)

        data = []
        for series in data_desc['series_list']:
            path = data_desc['data_root_dir'] + '/' + series['series']
            for episode in series['episodes']:
                full_path = path + '/' + episode['name']
                ground_truth = episode['ground_truth']
                if len(ground_truth) > 0:
                    episode_data = {'path': full_path, 'ground_truth': ground_truth}
                    data.append(episode_data)

        return data


def append_empty_data(video_shape: list, video_data: np.array, video_ground_truth: np.array,
                      audio_shape: list, audio_data: np.array, audio_ground_truth: np.array, frames: int):
    video_data_out = append_data(video_shape, video_data,
                                 [np.full(shape=video_shape, fill_value=0) for _ in range(frames)])
    video_ground_truth_out = np.append(video_ground_truth, np.full(shape=[frames, 1], fill_value=0))

    audio_data_out = append_data(audio_shape, audio_data,
                                 [np.full(shape=audio_shape, fill_value=0) for _ in range(frames)])
    audio_ground_truth_out = np.append(audio_ground_truth, np.full(shape=[frames, 1], fill_value=0))

    return [video_data_out, video_ground_truth_out, audio_data_out, audio_ground_truth_out]


def append_data(data_shape: list, current_data: np.array, data_to_be_appended: np.array):
    data_frame_size = reduce(lambda x, y: x * y, data_shape)
    data_out_lst = np.array(np.append(current_data, data_to_be_appended))
    data_shape_out = np.append(np.array(data_out_lst.size / data_frame_size), np.array(data_shape)).astype(int)
    return np.reshape(data_out_lst, data_shape_out)


def get_output_filename(input_filename):
    if not input_filename.endswith(".mp4"):
        return input_filename
    output_filename_build_list = input_filename.split(".mp4")
    output_filename_build_list[-1] = "_out.mp4"
    return "".join(output_filename_build_list)


def get_preproc_filename(filename):
    if not filename.endswith(".mp4"):
        return filename
    output_filename_build_list = filename.split(".mp4")
    output_filename_build_list[-1] = "_preproc.mp4"
    return "".join(output_filename_build_list)


def get_every_nth_element(array: np.array, n: float):
    if n <= 0:
        raise ValueError('n step has to be positive number')

    if math.floor(n) == n:
        return array[::n]

    # n is floating number
    result_array = []
    current_index = 0
    while current_index < len(array):
        result_array.append(array[math.floor(current_index)])
        current_index += n

    return np.asarray(result_array)
