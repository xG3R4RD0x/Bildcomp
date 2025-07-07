import unittest
import numpy as np

from pipeline.stages.decorrelation.strategy.prediction_strategy import (
    PredictionStrategy,
    find_best_mode_and_residuals_uint8,
    apply_prediction_block_uint8,
)
from pipeline.stages.decorrelation.decorrelation_stage import separate_yuv_compression
from pipeline.stages.decorrelation.strategy.decode_prediction_strategy import (
    DecodePredictionStrategy as dps,
)
from pipeline.stages.decorrelation.strategy.transformation_strategy import BlockDCT as bd


class TestDecorrelationStage(unittest.TestCase):
    def setUp(self):

        # load sample video
        self.input_data = np.fromfile(
            "test_videos/Sign_Irene_352x288.yuv", dtype=np.uint8
        )
        self.input_height = 288
        self.input_width = 352

    def test_yuv_separation(self):
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Extract the first frame from the input data
        first_frame = self.input_data[:frame_size]

        # Call separate_yuv with the first frame
        video = separate_yuv_compression(first_frame, self.input_width, self.input_height)

        # Validate Y, U, V shapes
        self.assertEqual(video[0].shape, (self.input_height, self.input_width))
        self.assertEqual(video[1].shape, (self.input_height//2, self.input_width//2))
        self.assertEqual(video[2].shape, (self.input_height//2, self.input_width//2))

    def test_first_frame_prediction(self):
        """
        Test the prediction strategy for the first frame of the input video.
        Prints the best modes for each block in the frame.
        """
        frame_size = self.input_width * self.input_height * 3 // 2
        first_frame = self.input_data[:frame_size]
        video = separate_yuv_compression(first_frame, self.input_width, self.input_height)
        # Usar PredictionStrategy
        prediction_strategy = PredictionStrategy()
        result = prediction_strategy.process(
            {"y": video[0], "u": video[1], "v": video[2]}, block_size=8
        )
        # Assert that the result contains expected keys
        self.assertIn("y_mode_flags", result)
        self.assertIsInstance(result["y_mode_flags"], np.ndarray)
    def test_first_frame_prediction_with_borders(self):
        """
        Test the prediction strategy for the first frame of the video but solo con los bordes del bloque
        """
        frame_size = self.input_width * self.input_height * 3 // 2
        first_frame = self.input_data[:frame_size]
        video = separate_yuv_compression(first_frame, self.input_width, self.input_height)
        prediction_strategy = PredictionStrategy()
        result = prediction_strategy.process(
            {"y": video[0], "u": video[1], "v": video[2]}, block_size=8, borders=True
        )


    def test_prediction_strategy(self):
        """
        Test the prediction strategy by processing all frames in the input video.
        Validates the residuals and mode flags for each block in each frame.
        """
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Initialize the prediction strategy
        prediction_strategy = PredictionStrategy()

        # Iterate over all frames in the input data
        num_frames = len(self.input_data) // frame_size
        for frame_idx in range(num_frames):
            # Extract the current frame from the input data
            start = frame_idx * frame_size
            end = start + frame_size
            frame_data = self.input_data[start:end]

            # Call separate_yuv with the current frame
            video = separate_yuv_compression(frame_data, self.input_width, self.input_height)

            # Call the prediction strategy process method
            result = prediction_strategy.process(
                {"y": video[0], "u": video[1], "v": video[2]}, block_size=8
            )

            # Validate the result structure
            self.assertIn("y_residual", result)
            self.assertIn("u_residual", result)
            self.assertIn("v_residual", result)
            self.assertIn("y_mode_flags", result)
            self.assertIn("u_mode_flags", result)
            self.assertIn("v_mode_flags", result)

            # Validate residuals and mode flags for each block
            # Validate residuals and mode flags for Y component
            residuals_y = result["y_residual"]
            mode_flags_y = result["y_mode_flags"]
            self.assertEqual(residuals_y.shape, video[0].shape)
            block_height = self.input_height // 8
            block_width = self.input_width // 8
            self.assertEqual(mode_flags_y.shape, (block_height, block_width))

            # Validate residuals and mode flags for U and V components
            for component in ["u", "v"]:
                residuals = result[f"{component}_residual"]
                mode_flags = result[f"{component}_mode_flags"]

                idx = {"u": 1, "v": 2}[component]
                self.assertEqual(residuals.shape, video[idx].shape)
                self.assertEqual(mode_flags.shape, (block_height//2, block_width//2))

            # print(f"Frame {frame_idx + 1}/{num_frames} processed successfully.")
    def test_prediction_strategy_with_borders(self):
        """
        Test the prediction strategy with borders for all the video frames.
        """
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Initialize the prediction strategy
        prediction_strategy = PredictionStrategy()

        # Iterate over all frames in the input data
        num_frames = len(self.input_data) // frame_size
        for frame_idx in range(num_frames):
            # Extract the current frame from the input data
            start = frame_idx * frame_size
            end = start + frame_size
            frame_data = self.input_data[start:end]

            # Call separate_yuv with the current frame
            video = separate_yuv_compression(frame_data, self.input_width, self.input_height)

            # Call the prediction strategy process method
            result = prediction_strategy.process(
                {"y": video[0], "u": video[1], "v": video[2]}, block_size=8, borders=True
            )

            # Validate the result structure
            self.assertIn("y_residual", result)
            self.assertIn("u_residual", result)
            self.assertIn("v_residual", result)
            self.assertIn("y_mode_flags", result)
            self.assertIn("u_mode_flags", result)
            self.assertIn("v_mode_flags", result)

            # Validate residuals and mode flags for each block
            # Validate residuals and mode flags for Y component
            residuals_y = result["y_residual"]
            mode_flags_y = result["y_mode_flags"]
            self.assertEqual(residuals_y.shape, video[0].shape)
            block_height = self.input_height // 8
            block_width = self.input_width // 8
            self.assertEqual(mode_flags_y.shape, (block_height, block_width))

            # Validate residuals and mode flags for U and V components
            for component in ["u", "v"]:
                residuals = result[f"{component}_residual"]
                mode_flags = result[f"{component}_mode_flags"]

                idx = {"u": 1, "v": 2}[component]
                self.assertEqual(residuals.shape, video[idx].shape)
                self.assertEqual(mode_flags.shape, (block_height // 2, block_width // 2))

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
        prediction_strategy = PredictionStrategy()
        decode_strategy = dps()
        video = separate_yuv_compression(first_frame, self.input_width, self.input_height)
        result = prediction_strategy.process(
            {"y": video[0], "u": video[1], "v": video[2]}, block_size=8
        )
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
        np.testing.assert_array_equal(decoded_y, video[0])
        np.testing.assert_array_equal(decoded_u, video[1])
        np.testing.assert_array_equal(decoded_v, video[2])

        # Verify that the decoded values are within the valid range
        self.assertTrue(np.all(decoded_y >= 0) and np.all(decoded_y <= 255))
        self.assertTrue(np.all(decoded_u >= 0) and np.all(decoded_u <= 255))
        self.assertTrue(np.all(decoded_v >= 0) and np.all(decoded_v <= 255))
        
    def test_encode_decode_first_frame_with_borders(self):
        """
        Test encoding and decoding of the first frame with borders.
        """
        # Calculate frame size for YUV420p
        frame_size = self.input_width * self.input_height * 3 // 2

        # Extract the first frame from the input data
        first_frame = self.input_data[:frame_size]

        # Initialize the prediction and decode strategies
        prediction_strategy = PredictionStrategy()
        decode_strategy = dps()
        video = separate_yuv_compression(first_frame, self.input_width, self.input_height)
        result = prediction_strategy.process(
            {"y": video[0], "u": video[1], "v": video[2]}, block_size=8, borders=True
        )
        decoded_y = decode_strategy.decode(
            result["y_residual"], result["y_mode_flags"], block_size=8, borders=True
        )
        decoded_u = decode_strategy.decode(
            result["u_residual"], result["u_mode_flags"], block_size=8, borders=True
        )
        decoded_v = decode_strategy.decode(
            result["v_residual"], result["v_mode_flags"], block_size=8, borders=True
        )

        # Verify that the decoded components match the original components
        np.testing.assert_array_equal(decoded_y, video[0])
        np.testing.assert_array_equal(decoded_u, video[1])
        np.testing.assert_array_equal(decoded_v, video[2])

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

        video = separate_yuv_compression(first_frame, self.input_width, self.input_height)
        original_y = video[0].astype(np.float64)

        dct = bd(width=self.input_width, height=self.input_height, block_size=8)

        transformed = dct.transform(original_y)
        reconstructed = dct.inverse_transform(transformed)

        # Validate that reconstruction is close to original
        np.testing.assert_array_almost_equal(reconstructed, original_y, decimal=5)
        self.assertFalse(np.array_equal(transformed, original_y))

    def test_prediction_and_decode_with_borders_on_small_array(self):
        """
        Test explícito de predicción y decodificación con bordes usando un array pequeño y valores conocidos.
        Imprime valores predichos, residuales y reconstruidos para depuración.
        """
        # Crear un array 10x10 con valores crecientes
        arr = np.arange(100, dtype=np.uint8).reshape((10, 10))
        video = {"y": arr, "u": arr, "v": arr}
        block_size = 4
        prediction_strategy = PredictionStrategy()
        decode_strategy = dps()
        result = prediction_strategy.process({"y": arr, "u": arr, "v": arr}, block_size=block_size, borders=True)
        residuals = result["y_residual"]
        mode_flags = result["y_mode_flags"]
        decoded = decode_strategy.decode(residuals, mode_flags, block_size=block_size, borders=True)

        # Imprimir diferencias si las hay
        if not np.array_equal(decoded, arr):
            print("Diferencias encontradas en la decodificación con bordes:")
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if decoded[i, j] != arr[i, j]:
                        print(f"Posición ({i},{j}): original={arr[i, j]}, decodificado={decoded[i, j]}, residual={residuals[i, j]}")
        else:
            print("La decodificación con bordes es correcta para el array de prueba.")
        # Asegura que la decodificación sea correcta
        np.testing.assert_array_equal(decoded, arr)

if __name__ == "__main__":
    unittest.main()
