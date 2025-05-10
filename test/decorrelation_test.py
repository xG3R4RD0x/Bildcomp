import unittest
import numpy as np

from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage as ds


class TestDecorrelationStage(unittest.TestCase):
    def setUp(self):

        # load sample video
        self.input_data = np.fromfile("carphone_176x144.yuv", dtype=np.uint8)
        self.input_height = 144
        self.input_width = 176

    def test_yuv_separation(self):
        video = ds.separate_yuv(self.input_data, self.input_height, self.input_width)
        print(video)


if __name__ == "__main__":
    unittest.main()
