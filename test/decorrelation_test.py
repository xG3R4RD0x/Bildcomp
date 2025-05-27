import unittest
import numpy as np

from pipeline.stages.decorrelation.strategy.prediction_strategy import (
    PredictionStrategy as ps,
)
from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage as ds
from pipeline.stages.decorrelation.strategy.decode_prediction_strategy import (
    DecodePredictionStrategy as dps,
)
from pipeline.stages.decorrelation.strategy.transformation_strategy import BlockDCT as bd


class TestDecorrelationStage(unittest.TestCase):
    def setUp(self):

        # load sample video
        self.input_data = np.fromfile(
            "test_videos/carphone_176x144.yuv", dtype=np.uint8
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

    def test_first_frame_prediction(self):
        """
        Test the prediction strategy for the first frame of the input video.
        Prints the best modes for each block in the frame.
        """
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Extract the first frame from the input data
        first_frame = self.input_data[:frame_size]

        # Initialize the prediction strategy
        prediction_strategy = ps()

        # Call separate_yuv with the first frame
        video = ds.separate_yuv(first_frame, self.input_width, self.input_height)

        # Call the prediction strategy process method
        result = prediction_strategy.process(
            {"y": video["y"], "u": video["u"], "v": video["v"]}, block_size=8
        )

        # Print the best modes for each block
        # for component in ["y", "u", "v"]:
        #     mode_flags = result[f"{component}_mode_flags"]
        #     print(f"Best modes for {component.upper()} component:")
        #     print(mode_flags)

    def test_prediction_strategy(self):
        """
        Test the prediction strategy by processing all frames in the input video.
        Validates the residuals and mode flags for each block in each frame.
        """
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Initialize the prediction strategy
        prediction_strategy = ps()

        # Iterate over all frames in the input data
        num_frames = len(self.input_data) // frame_size
        for frame_idx in range(num_frames):
            # Extract the current frame from the input data
            start = frame_idx * frame_size
            end = start + frame_size
            frame_data = self.input_data[start:end]

            # Call separate_yuv with the current frame
            video = ds.separate_yuv(frame_data, self.input_width, self.input_height)

            # Call the prediction strategy process method
            result = prediction_strategy.process(
                {"y": video["y"], "u": video["u"], "v": video["v"]}, block_size=8
            )

            # Validate the result structure
            self.assertIn("y_residual", result)
            self.assertIn("u_residual", result)
            self.assertIn("v_residual", result)
            self.assertIn("y_mode_flags", result)
            self.assertIn("u_mode_flags", result)
            self.assertIn("v_mode_flags", result)

            # Validate residuals and mode flags for each block
            for component in ["y", "u", "v"]:
                residuals = result[f"{component}_residual"]
                mode_flags = result[f"{component}_mode_flags"]

                # Check that residuals have the same shape as the original component
                self.assertEqual(residuals.shape, video[component].shape)

                # Check that mode flags have the correct shape for blocks
                block_height = self.input_height // 8
                block_width = self.input_width // 8
                self.assertEqual(mode_flags.shape, (block_height, block_width))

            # print(f"Frame {frame_idx + 1}/{num_frames} processed successfully.")

    def test_encode_decode_first_frame(self):
        """
        Test encoding and decoding of the first frame to ensure the decoded frame matches the original.
        """
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Extract the first frame from the input data
        first_frame = self.input_data[:frame_size]

        # Initialize the prediction and decode strategies
        prediction_strategy = ps()
        decode_strategy = dps()

        # Call separate_yuv with the first frame
        video = ds.separate_yuv(first_frame, self.input_width, self.input_height)

        # Encode the frame using the prediction strategy
        result = prediction_strategy.process(
            {"y": video["y"], "u": video["u"], "v": video["v"]}, block_size=8
        )

        # Decode the frame using the decode strategy
        decoded_y = decode_strategy.decode(
            result["y_residual"], result["y_mode_flags"], block_size=8
        )
        decoded_u = decode_strategy.decode(
            result["u_residual"], result["u_mode_flags"], block_size=8
        )
        decoded_v = decode_strategy.decode(
            result["v_residual"], result["v_mode_flags"], block_size=8
        )

        # Verify that the decoded components match the original components
        np.testing.assert_array_equal(decoded_y, video["y"])
        np.testing.assert_array_equal(decoded_u, video["u"])
        np.testing.assert_array_equal(decoded_v, video["v"])

        # Verify that the decoded values are within the valid range
        self.assertTrue(np.all(decoded_y >= 0) and np.all(decoded_y <= 255))
        self.assertTrue(np.all(decoded_u >= 0) and np.all(decoded_u <= 255))
        self.assertTrue(np.all(decoded_v >= 0) and np.all(decoded_v <= 255))

    def test_block_dct_on_yuv_first_frame(self):
        """
        Test the BlockDCT by applying DCT and inverse DCT on the first frame of a YUV video.
        """
        # Get first frame size (Y plane only) for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2
        first_frame = self.input_data[:frame_size]

        video = ds.separate_yuv(first_frame, self.input_width, self.input_height)
        original_y = video["y"].astype(np.float64)

        dct = bd(width=self.input_width, height=self.input_height, block_size=8)

        transformed = dct.transform(original_y)
        reconstructed = dct.inverse_transform(transformed)

        # Validate that reconstruction is close to original
        np.testing.assert_array_almost_equal(reconstructed, original_y, decimal=5)
        self.assertFalse(np.array_equal(transformed, original_y))

if __name__ == "__main__":
    unittest.main()
