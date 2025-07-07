from typing import Any, Dict
from pipeline.interfaces.base_stage import CompressionStage
import numpy as np
from pipeline.stages.decorrelation.strategy.prediction_strategy import (
    PredictionStrategy as ps,
)
from numba import njit

# from .strategy.prediction_strategy import PredictionStrategy
# from .strategy.transformation_strategy import TransformationStrategy
# from .compression_stage import CompressionStage


class DecorrelationStage(CompressionStage):
    def __init__(self):
        self.prediction_strategy = ps()

    def name(self) -> str:
        return "Decorrelation Stage"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decorrelation process:
        first we use prediction
        then we use transformation
        """

        predicted_data = self.prediction_strategy.process(data)

        # transformed_data = self.transformation_strategy.process(predicted_data)

        # return transformed_data

    def separate_yuv(self,
        yuv_data: np.ndarray, width: int, height: int
    ) -> Dict[str, np.ndarray]:
        """
        Separates the YUV data into individual Y, U, and V components for a YUV 420p video,
        upscales U and V, and converts to RGB.

        :param yuv_data: The YUV video frame data as a numpy array.
        :param width: The width of the video frame.
        :param height: The height of the video frame.
        :return: A dictionary containing Y, U, V, and RGB components as separate numpy arrays.
        """
        # Calculate sizes
        y_size = width * height
        uv_size = (width // 2) * (height // 2)
        expected_size = y_size + 2 * uv_size

        # Validate input size
        if len(yuv_data) != expected_size:
            raise ValueError(
                f"Invalid YUV data size. Expected {expected_size}, got {len(yuv_data)}"
            )

        # Split the Y, U, and V components
        y = np.frombuffer(yuv_data[0:y_size], dtype=np.uint8).reshape((height, width))
        u = np.frombuffer(yuv_data[y_size : y_size + uv_size], dtype=np.uint8).reshape(
            (height // 2, width // 2)
        )
        v = np.frombuffer(yuv_data[y_size + uv_size :], dtype=np.uint8).reshape(
            (height // 2, width // 2)
        )

        # Upscale U and V to match Y dimensions
        u_up = u.repeat(2, axis=0).repeat(2, axis=1)
        v_up = v.repeat(2, axis=0).repeat(2, axis=1)

        # Convert YUV to RGB
        yuv = np.stack((y, u_up, v_up), axis=2).astype(np.float32)
        yuv[:, :, 0] -= 16
        yuv[:, :, 1] -= 128
        yuv[:, :, 2] -= 128

        m = np.array([[1.0, 0.0, 1.402], [1.0, -0.34414, -0.71414], [1.0, 1.772, 0.0]])
        rgb = yuv @ m.T
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Return the components in a dictionary
        return {"y": y, "u": u_up, "v": v_up, "rgb": rgb}

    def separate_yuv_compression(self,
        yuv_data: np.ndarray, width: int, height: int
    ) -> Dict[str, np.ndarray]:
        """
        Separates the YUV data into individual Y, U, and V components for a YUV 420p video,
       
        :param yuv_data: The YUV video frame data as a numpy array.
        :param width: The width of the video frame.
        :param height: The height of the video frame.
        :return: A dictionary containing Y, U, V as separate numpy arrays.
        """
        # Calculate sizes
        y_size = width * height
        uv_size = (width // 2) * (height // 2)
        expected_size = y_size + 2 * uv_size

        # Validate input size
        if len(yuv_data) != expected_size:
            raise ValueError(
                f"Invalid YUV data size. Expected {expected_size}, got {len(yuv_data)}"
            )

        # Split the Y, U, and V components
        y = np.frombuffer(yuv_data[0:y_size], dtype=np.uint8).reshape((height, width))
        u = np.frombuffer(yuv_data[y_size : y_size + uv_size], dtype=np.uint8).reshape(
            (height // 2, width // 2)
        )
        v = np.frombuffer(yuv_data[y_size + uv_size :], dtype=np.uint8).reshape(
            (height // 2, width // 2)
        )

        # Return the components in a dictionary
        return {"y": y, "u": u, "v": v}
    
@njit
def separate_yuv_compression(
        yuv_data: np.ndarray, width: int, height: int
    ) -> dict:
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    expected_size = y_size + 2 * uv_size

    if yuv_data.size != expected_size:
        raise ValueError("Invalid YUV data size.")

    y = yuv_data[0:y_size].reshape((height, width))
    u = yuv_data[y_size : y_size + uv_size].reshape((height // 2, width // 2))
    v = yuv_data[y_size + uv_size :].reshape((height // 2, width // 2))

    # Numba: solo claves int
    return {0: y, 1: u, 2: v}

