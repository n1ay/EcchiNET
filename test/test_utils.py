import unittest
from utils import utils
import numpy as np
from numpy.testing import assert_array_equal


class TestUtils(unittest.TestCase):

    def test_preprocess_lstm_context(self):
        examples = [
            (np.asarray([]), np.asarray([]), 3, 5),
            (np.asarray([]), np.asarray([]), 0, 6),
            (np.asarray([1, 2, 3, 4, 5, 6, 7]), np.asarray([[1], [2], [3], [4], [5], [6], [7]]), 0, 0),
            (np.asarray([1, 2, 3, 4, 5, 6, 7]), np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
                                                            [5, 6, 7], [6, 7, 7], [7, 7, 7]]), 0, 2),
            (np.asarray([1, 2, 3, 4, 5, 6, 7]), np.asarray([[1, 1, 1], [1, 1, 2], [1, 2, 3], [2, 3, 4],
                                                            [3, 4, 5], [4, 5, 6], [5, 6, 7]]), 2, 0),
            (np.asarray([1, 2, 3, 4, 5, 6, 7]), np.asarray([[1, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
                                                            [4, 5, 6], [5, 6, 7], [6, 7, 7]]), 1, 1),
        ]

        for data in examples:
            input_data = data[0]
            output_true = data[1]
            backward_time_step = data[2]
            forward_time_step = data[3]
            assert_array_equal(output_true, utils.preprocess_lstm_context(input_data, backward_time_step, forward_time_step))

    def test_reshape_prune_extra(self):
        examples = [
            (np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]), np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), (2, 2, 2)),
            (np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]), np.asarray([[[1, 2, 3], [4, 5, 6]]]), (1, 2, 3)),
            (np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]), np.asarray([[[[1], [2]], [[3], [4]]]]), (1, 2, 2, 1))
        ]

        for data in examples:
            input_data = data[0]
            output_true = data[1]
            dst_shape = data[2]
            assert_array_equal(output_true, utils.reshape_prune_extra(input_data, dst_shape))
