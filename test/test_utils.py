import unittest
from utils import utils
import numpy as np
from numpy.testing import assert_array_equal


class TestUtils(unittest.TestCase):

    def test_preprocess_lstm_context(self):
        examples = [
            {
                'input_data': np.asarray([]),
                'output_true': np.asarray([]),
                'backward_time_step': 3,
                'forward_time_step': 5
            },
            {
                'input_data': np.asarray([]),
                'output_true': np.asarray([]),
                'backward_time_step': 0,
                'forward_time_step': 6
            },
            {
                'input_data': np.asarray([1, 2, 3, 4, 5, 6, 7]),
                'output_true': np.asarray([[1], [2], [3], [4], [5], [6], [7]]),
                'backward_time_step': 0,
                'forward_time_step': 0
            },
            {
                'input_data': np.asarray([1, 2, 3, 4, 5, 6, 7]),
                'output_true': np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
                                           [5, 6, 7], [6, 7, 7], [7, 7, 7]]),
                'backward_time_step': 0,
                'forward_time_step': 2
            },
            {
                'input_data': np.asarray([1, 2, 3, 4, 5, 6, 7]),
                'output_true': np.asarray([[1, 1, 1], [1, 1, 2], [1, 2, 3], [2, 3, 4],
                                           [3, 4, 5], [4, 5, 6], [5, 6, 7]]),
                'backward_time_step': 2,
                'forward_time_step': 0
            },
            {
                'input_data': np.asarray([1, 2, 3, 4, 5, 6, 7]),
                'output_true': np.asarray([[1, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
                                           [4, 5, 6], [5, 6, 7], [6, 7, 7]]),
                'backward_time_step': 1,
                'forward_time_step': 1
            },
        ]

        for data in examples:
            input_data = data['input_data']
            output_true = data['output_true']
            backward_time_step = data['backward_time_step']
            forward_time_step = data['forward_time_step']
            assert_array_equal(output_true,
                               utils.preprocess_lstm_context(input_data, backward_time_step, forward_time_step))

    def test_reshape_prune_extra(self):
        examples = [
            {
                'input_data': np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]),
                'output_true': np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                'dst_shape': (2, 2, 2)
            },
            {
                'input_data': np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]),
                'output_true': np.asarray([[[1, 2, 3], [4, 5, 6]]]),
                'dst_shape': (1, 2, 3)
            },
            {
                'input_data': np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]),
                'output_true': np.asarray([[[[1], [2]], [[3], [4]]]]),
                'dst_shape': (1, 2, 2, 1)
            }
        ]

        for data in examples:
            input_data = data['input_data']
            output_true = data['output_true']
            dst_shape = data['dst_shape']
            assert_array_equal(output_true, utils.reshape_prune_extra(input_data, dst_shape))

    def test_generate_ground_truth_from_frames(self):
        examples = [
            {
                'preproc_frames_num': 20,
                'true_marked_frame_ranges': [(0, 2), (3, 5), (6, 8), (10, 11), (13, 15), (16, 16)],
                'output_true': np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0])
            },
            {
                'preproc_frames_num': 5,
                'true_marked_frame_ranges': [],
                'output_true': np.asarray([0, 0, 0, 0, 0])
            },
            {
                'preproc_frames_num': 6,
                'true_marked_frame_ranges': [(0, 5)],
                'output_true': np.asarray([1, 1, 1, 1, 1, 1])
            }
        ]

        for data in examples:
            preproc_frames_num = data['preproc_frames_num']
            true_marked_frame_ranges = data['true_marked_frame_ranges']
            output_true = data['output_true']
            assert_array_equal(output_true, utils.generate_ground_truth_from_frames(preproc_frames_num,
                                                                                    true_marked_frame_ranges))

    def test_generate_time_frames_from_binary_vec(self):
        examples = [
            {
                'preproc_fps': 1,
                'video_padding_time': 1,
                'positive_threshold': 0,
                'binary_vec': np.asarray([]),
                'output_true': []
            },
            {
                'preproc_fps': 1,
                'video_padding_time': 1,
                'positive_threshold': 0,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 9)]
            },
            {
                'preproc_fps': 1,
                'video_padding_time': 1,
                'positive_threshold': 0.5,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 3), (8, 9)]
            },
            {
                'preproc_fps': 1,
                'video_padding_time': 1,
                'positive_threshold': 0.5,
                'binary_vec': np.asarray([0.5, 0.6, 0.6, 0.4, 0.3, 0.3, 0.4, 0.4, 0.2, 0.7]),
                'output_true': [(0, 3), (8, 9)]
            },
            {
                'preproc_fps': 2,
                'video_padding_time': 1,
                'positive_threshold': 0,
                'binary_vec': np.asarray([]),
                'output_true': []
            },
            {
                'preproc_fps': 2,
                'video_padding_time': 1,
                'positive_threshold': 0,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 4)]
            },
            {
                'preproc_fps': 2,
                'video_padding_time': 1,
                'positive_threshold': 0.5,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 2), (3, 4)]
            },
            {
                'preproc_fps': 2,
                'video_padding_time': 1,
                'positive_threshold': 0.5,
                'binary_vec': np.asarray([0.5, 0.6, 0.6, 0.4, 0.3, 0.3, 0.4, 0.4, 0.2, 0.7]),
                'output_true': [(0, 2), (3, 4)]
            },
            {
                'preproc_fps': 1,
                'video_padding_time': 3,
                'positive_threshold': 0,
                'binary_vec': np.asarray([]),
                'output_true': []
            },
            {
                'preproc_fps': 1,
                'video_padding_time': 3,
                'positive_threshold': 0,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 9)]
            },
            {
                'preproc_fps': 1,
                'video_padding_time': 3,
                'positive_threshold': 0.5,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 5), (6, 9)]
            },
            {
                'preproc_fps': 1,
                'video_padding_time': 3,
                'positive_threshold': 0.5,
                'binary_vec': np.asarray([0.5, 0.6, 0.6, 0.4, 0.3, 0.3, 0.4, 0.4, 0.2, 0.7]),
                'output_true': [(0, 5), (6, 9)]
            },
            {
                'preproc_fps': 2,
                'video_padding_time': 3,
                'positive_threshold': 0,
                'binary_vec': np.asarray([]),
                'output_true': []
            },
            {
                'preproc_fps': 2,
                'video_padding_time': 3,
                'positive_threshold': 0,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 4)]
            },
            {
                'preproc_fps': 2,
                'video_padding_time': 3,
                'positive_threshold': 0.5,
                'binary_vec': np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
                'output_true': [(0, 4)]
            },
            {
               'preproc_fps': 2,
               'video_padding_time': 3,
               'positive_threshold': 0.5,
               'binary_vec': np.asarray([0.5, 0.6, 0.6, 0.4, 0.3, 0.3, 0.4, 0.4, 0.2, 0.7]),
               'output_true': [(0, 4)]
            },
        ]

        for data in examples:
            preproc_fps = data['preproc_fps']
            video_padding_time = data['video_padding_time']
            binary_vec = data['binary_vec']
            positive_threshold = data['positive_threshold']
            output_true = data['output_true']
            assert_array_equal(output_true, utils.generate_time_frames_from_binary_vec(preproc_fps, video_padding_time, binary_vec, positive_threshold))

    def test_reshape_average_prune_extra(self):
        examples = [
            {
                'vector': np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                'dst_length': 4,
                'output_true': np.asarray([0.5, 0.5, 0.5, 0.5])
            },
            {
                'vector': np.array([0, 1, 4, 5, 9, 5, 6, 7]),
                'dst_length': 3,
                'output_true': np.asarray([0.5, 6, 5.5])
            },
            {
                'vector': np.array([0, 1, 4, 5, 9, 5, 6, 7]),
                'dst_length': 0,
                'output_true': np.asarray([])
            },
        ]

        for data in examples:
            vector = data['vector']
            dst_length = data['dst_length']
            output_true = data['output_true']
            assert_array_equal(output_true, utils.reshape_average_prune_extra(vector, dst_length))

    def test_load_parse_data_description(self):
        examples = [
            {
                'test_filename': 'data/test_data.json',
                'output_true': [
                    {
                        'path': 'data/ecchi_series/1.mp4',
                        'ground_truth': [[3544, 3643], [3649, 3749]]
                    },
                    {
                        'path': 'data/ecchi_series/2.mp4',
                        'ground_truth': [[2954, 3594], [4729, 5134]]
                    },
                    {
                        'path': 'data/ecchi_series/3.mp4',
                        'ground_truth': [[4659, 4818]]
                    }
                ]
            }
        ]

        for data in examples:
            test_filename = data['test_filename']
            output_true = data['output_true']
            self.assertEqual(output_true, utils.load_parse_data_description(test_filename))
