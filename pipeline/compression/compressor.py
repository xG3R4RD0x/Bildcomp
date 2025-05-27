import os
import numpy as np
import pickle
from typing import Dict, Any
from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage
from pipeline.stages.decorrelation.strategy.prediction_strategy import (
    PredictionStrategy,
)
from pipeline.stages.decorrelation.strategy.transformation_strategy import BlockDCT as bd


class Compressor:
    """
    Handles video compression using prediction-based decorrelation.
    Compresses YUV video files and stores them in the specified output directory.
    """

    def __init__(self):
        self.prediction_strategy = PredictionStrategy()

    def compress_video(
        self,
        height: int,
        width: int,
        input_path: str,
        output_dir: str = "test_videos/compressed/",
    ) -> str:
        """
        Compresses a YUV video file and saves it to the specified output directory.

        Args:
            input_path: Path to the input YUV video file
            output_dir: Directory to save the compressed file (default: test_videos/compressed/)

        Returns:
            Path to the compressed file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.decp")

        # Read and process the YUV file
        compressed_data = self._process_yuv_file(input_path, width, height)

        # Save compressed data
        with open(output_path, "wb") as f:
            pickle.dump(compressed_data, f)

        print(f"Compressed video saved to {output_path}")
        return output_path
    
    def compress_video_dct(
        self,
        height: int,
        width: int,
        input_path: str,
        output_dir: str = "test_videos/compressed/",
    ) -> str:
        """
        Compresses a YUV video file and saves it to the specified output directory.

        Args:
            input_path: Path to the input YUV video file
            output_dir: Directory to save the compressed file (default: test_videos/compressed/)

        Returns:
            Path to the compressed file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.dect")

        # Read and process the YUV file
        compressed_data = self._process_yuv_file_with_dct(input_path, width, height)

        # Save compressed data
        with open(output_path, "wb") as f:
            pickle.dump(compressed_data, f)

        print(f"Compressed video saved to {output_path}")
        return output_path

    def _process_yuv_file(self, path: str, width: int, height: int) -> Dict[str, Any]:
        """
        Processes a YUV file by reading frames and applying prediction.

        Args:
            path: Path to the YUV file
            width: Width of the video
            height: Height of the video

        Returns:
            A dictionary containing compressed video data
        """
        # Calculate frame size
        frame_size = width * height * 3 // 2  # YUV420p

        # Read the entire file
        with open(path, "rb") as f:
            data = f.read()

        # Calculate number of frames
        num_frames = len(data) // frame_size

        # Process frames
        compressed_frames = []

        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size

            # Get frame data
            frame_data = data[start_idx:end_idx]

            # Separate YUV components
            components = DecorrelationStage.separate_yuv(frame_data, width, height)

            # Apply prediction strategy
            frame_result = self.prediction_strategy.process(
                {"y": components["y"], "u": components["u"], "v": components["v"]},
                block_size=8,
            )

            # Store compressed frame data
            compressed_frames.append(
                {
                    "y_residual": frame_result["y_residual"],
                    "u_residual": frame_result["u_residual"],
                    "v_residual": frame_result["v_residual"],
                    "y_mode_flags": frame_result["y_mode_flags"],
                    "u_mode_flags": frame_result["u_mode_flags"],
                    "v_mode_flags": frame_result["v_mode_flags"],
                }
            )

        # Prepare compressed data structure
        compressed_data = {
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frames": compressed_frames,
        }

        return compressed_data
    
    def _process_yuv_file_with_dct(
        self, path: str, width: int, height: int
    ) -> Dict[str, Any]:
        """
        Processes a YUV file by reading frames and applying dct transformation.

        Args:
            path: Path to the YUV file
            width: Width of the video
            height: Height of the video

        Returns:
            A dictionary containing compressed video data
        """
        # Calculate frame size
        frame_size = width * height * 3 // 2  # YUV420p

        # Read the entire file
        with open(path, "rb") as f:
            data = f.read()

        # Calculate number of frames
        num_frames = len(data) // frame_size

        # Process frames
        compressed_frames = []

        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size

            # Get frame data
            frame_data = data[start_idx:end_idx]

            # Separate YUV components
            components = DecorrelationStage.separate_yuv(frame_data, width, height)

            original_y = components["y"].astype(np.float64)
            original_u = components["u"].astype(np.float64)
            original_v = components["v"].astype(np.float64)
            dct = bd(width=width, height=height, block_size=8)
            # Apply transformation strategy
            frame_result_y = dct.transform(original_y)
            frame_result_u = dct.transform(original_u)
            frame_result_v = dct.transform(original_v)
            # Store compressed frame data
            compressed_frames.append(
                {
                    "y_residual": frame_result_y,
                    "u_residual": frame_result_u,
                    "v_residual": frame_result_v}
            )

        # Prepare compressed data structure
        compressed_data = {
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frames": compressed_frames,
        }

        return compressed_data
        
        
        

    def decompress_video(self, compressed_path: str, output_path: str = None) -> str:
        """
        Decompress a previously compressed video file.

        Args:
            compressed_path: Path to the compressed file
            output_path: Path to save the decompressed YUV file (optional)

        Returns:
            Path to the decompressed file
        """
        # Create output directory if it doesn't exist
        output_dir = "test_videos/decompressed/"
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename if not provided
        if output_path is None:
            filename = os.path.basename(compressed_path)
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.yuv"
            )

        # Load compressed data
        with open(compressed_path, "rb") as f:
            compressed_data = pickle.load(f)

        # Extract video parameters
        width = compressed_data["width"]
        height = compressed_data["height"]
        num_frames = compressed_data["num_frames"]

        # Create a DecodePredictionStrategy instance
        from pipeline.stages.decorrelation.strategy.decode_prediction_strategy import (
            DecodePredictionStrategy,
        )

        decoder = DecodePredictionStrategy()

        # Open output file for writing
        with open(output_path, "wb") as f:
            # Process each frame
            for frame_data in compressed_data["frames"]:
                # Decode Y, U, and V components
                y_decoded = decoder.decode(
                    frame_data["y_residual"], frame_data["y_mode_flags"], block_size=8
                )
                u_decoded = decoder.decode(
                    frame_data["u_residual"], frame_data["u_mode_flags"], block_size=8
                )
                v_decoded = decoder.decode(
                    frame_data["v_residual"], frame_data["v_mode_flags"], block_size=8
                )

                # Convert the upscaled U and V components back to their original size for YUV420 format
                u_downscaled = u_decoded[::2, ::2]
                v_downscaled = v_decoded[::2, ::2]

                # Write YUV components to file
                f.write(y_decoded.tobytes())
                f.write(u_downscaled.tobytes())
                f.write(v_downscaled.tobytes())

        print(f"Decompressed video saved to {output_path}")
        return output_path
    
    def decompress_video_with_dct(self, compressed_path: str, output_path: str = None) -> str:
        """
        Decompress a previously compressed video file with DCT.

        Args:
            compressed_path: Path to the compressed file
            output_path: Path to save the decompressed YUV file (optional)

        Returns:
            Path to the decompressed file
        """
        # Create output directory if it doesn't exist
        output_dir = "test_videos/decompressed/"
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename if not provided
        if output_path is None:
            filename = os.path.basename(compressed_path)
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.yuv"
            )

        # Load compressed data
        with open(compressed_path, "rb") as f:
            compressed_data = pickle.load(f)

        # Extract video parameters
        width = compressed_data["width"]
        height = compressed_data["height"]
        num_frames = compressed_data["num_frames"]

        # Initialize BlockDCT instance
        from pipeline.stages.decorrelation.strategy.transformation_strategy import BlockDCT
        dct = BlockDCT(width=width, height=height, block_size=8)

        # Open output file for writing
        with open(output_path, "wb") as f:
            # Process each frame
            for frame_data in compressed_data["frames"]:
                # Decode Y, U, and V components using inverse DCT
                y_decoded = dct.inverse_transform(frame_data["y_residual"])
                u_decoded = dct.inverse_transform(frame_data["u_residual"])
                v_decoded = dct.inverse_transform(frame_data["v_residual"])

                # Convert the upscaled U and V components back to their original size for YUV420 format
                u_downscaled = u_decoded[::2, ::2]
                v_downscaled = v_decoded[::2, ::2]

                # Write YUV components to file
                f.write(y_decoded.astype(np.uint8).tobytes())
                f.write(u_downscaled.astype(np.uint8).tobytes())
                f.write(v_downscaled.astype(np.uint8).tobytes())

        print(f"Decompressed video saved to {output_path}")
        return output_path
