import unittest
import numpy as np

from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage as ds


class TestDecorrelationStage(unittest.TestCase):
    def setUp(self):

        # load sample video
        self.input_data = np.fromfile(
            "test_videos\carphone_176x144.yuv", dtype=np.uint8
        )
        self.input_height = 144
        self.input_width = 176

    def test_yuv_separation(self):
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Extract the first frame from the input data
        first_frame = self.input_data[:frame_size]

        # Call separate_yuv with the first frame
        video = ds.separate_yuv(first_frame, self.input_width, self.input_height)

        # Validate Y, U, V shapes
        self.assertEqual(video["y"].shape, (self.input_height, self.input_width))
        self.assertEqual(video["u"].shape, (self.input_height, self.input_width))
        self.assertEqual(video["v"].shape, (self.input_height, self.input_width))

        # Validate RGB shape
        self.assertEqual(video["rgb"].shape, (self.input_height, self.input_width, 3))

        # Validate RGB values are within range
        self.assertTrue(np.all(video["rgb"] >= 0) and np.all(video["rgb"] <= 255))


if __name__ == "__main__":
    unittest.main()
