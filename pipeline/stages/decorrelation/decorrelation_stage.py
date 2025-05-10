from typing import Any, Dict
from pipeline.interfaces.base_stage import CompressionStage
import numpy as np

# from .strategy.prediction_strategy import PredictionStrategy
# from .strategy.transformation_strategy import TransformationStrategy
# from .compression_stage import CompressionStage


class DecorrelationStage(CompressionStage):
    # def __init__(
    #     self,
    #     # prediction_strategy: PredictionStrategy,
    #     # transformation_strategy: TransformationStrategy,
    # ):
    # self.prediction_strategy = prediction_strategy
    # self.transformation_strategy = transformation_strategy

    def name(self) -> str:
        return "Decorrelation Stage"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decorrelation process:
        first we use prediction
        then we use transformation
        """
        # predicted_data = self.prediction_strategy.process(data)

        # transformed_data = self.transformation_strategy.process(predicted_data)

        # return transformed_data

    def separate_yuv(
        yuv_data: np.ndarray, width: int, height: int
    ) -> Dict[str, np.ndarray]:
        """
        Separates the YUV data into individual Y, U, and V components for a YUV 420p video.

        :param yuv_data: The YUV video frame data as a numpy array.
        :param width: The width of the video frame.
        :param height: The height of the video frame.
        :return: A dictionary containing Y, U, and V components as separate numpy arrays.
        """
        # Number of pixels in the Y component (luminance)
        y_size = width * height

        # U and V components are subsampled by a factor of 2 in each dimension (for YUV 420p)
        uv_width = width // 2
        uv_height = height // 2
        uv_size = uv_width * uv_height

        # Split the Y, U, and V components
        y = yuv_data[:y_size].reshape((height, width))  # Y component
        u = yuv_data[y_size : y_size + uv_size].reshape(
            (uv_height, uv_width)
        )  # U component
        v = yuv_data[y_size + uv_size :].reshape((uv_height, uv_width))  # V component

        # Return the components in a dictionary
        return {"y": y, "u": u, "v": v}
