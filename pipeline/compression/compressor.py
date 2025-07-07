import os
import numpy as np
import pickle
import struct
from typing import Dict, Any
from pipeline.stages.decorrelation.decorrelation_stage import separate_yuv_compression
from pipeline.stages.decorrelation.strategy.prediction_strategy import (
    PredictionStrategy,
)
from pipeline.stages.decorrelation.strategy.transformation_strategy import (
    BlockDCT as bd,
)
from pipeline.stages.quantization.quantization_stage import QuantizationStage as qs

from pipeline.stages.decorrelation.strategy.decode_prediction_strategy import (
            DecodePredictionStrategy,
        )


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
        borders: bool = False,
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
        compressed_data = self._process_yuv_file(input_path, width, height, borders=borders)

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
        Compresses a YUV video file and saves it to the specified output directory in a compact binary format.

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

        # Save compressed data in compact binary format
        with open(output_path, "wb") as f:
            # Write header: width, height, num_frames (all int32)
            f.write(struct.pack("iii", width, height, compressed_data["num_frames"]))
            for frame in compressed_data["frames"]:
                # Write shapes for each channel (y, u, v)
                for key in ["y_residual", "u_residual", "v_residual"]:
                    arr = frame[key].astype(np.float32)
                    # Write shape (2 int32: rows, cols)
                    f.write(struct.pack("ii", *arr.shape))
                    # Write data
                    f.write(arr.tobytes())

        print(f"Compressed video saved to {output_path}")
        return output_path

    def compress_video_dct_and_quant(
        self,
        height: int,
        width: int,
        input_path: str,
        output_dir: str = "test_videos/compressed/",
        block_size: int = 8,
        levels: int = 128,
    ) -> str:
        """
        Compresses a YUV video file and saves it to the specified output directory in a compact binary format.

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
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.dectq")

        # Read and process the YUV file
        compressed_data = self._process_yuv_file_with_dct_and_quant(input_path, width, height)

        # Save compressed data in compact binary format
        with open(output_path, "wb") as f:
            # Write header: width, height, num_frames, block_size, levels (all int32)
            f.write(struct.pack("iiiii", width, height, compressed_data["num_frames"], block_size, levels))
            for frame in compressed_data["frames"]:
                # Write shapes for each channel (y, u, v)
                for key in ["y_residual", "u_residual", "v_residual"]:
                    arr = frame[key]
                    min_val = frame[key[0] + "_min"]
                    step = frame[key[0] + "_step"]
                   
                    # Guardar shape (2 int32), min (float32), step (float32)
                    f.write(struct.pack("ii", *arr.shape))
                    f.write(struct.pack("ff", float(min_val), float(step)))
                    f.write(arr.tobytes())

        print(f"Compressed video saved to {output_path}")
        return output_path

    def _process_yuv_file(self, path: str, width: int, height: int, borders: bool=False) -> Dict[str, Any]:
        """
        Processes a YUV file by reading frames and applying prediction.

        Args:
            path: Path to the YUV file
            width: Width of the video
            height: Height of the video
            borders: Whether to use border pixels for prediction

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
            frame_data = np.frombuffer(data[start_idx:end_idx], dtype=np.uint8)

            # Separate YUV components (returns a tuple/list of 3 arrays: [Y, U, V])
            components = separate_yuv_compression(frame_data, width, height)

            # Apply prediction strategy using array indices for YUV
            frame_result = self.prediction_strategy.process(
                {0: components[0], 1: components[1], 2: components[2]},
                block_size=8, borders=borders
            )

            # Store compressed frame data using indices instead of string keys
            compressed_frames.append(
                {
                    0: {
                        "residual": frame_result[0]["residual"],
                        "mode_flags": frame_result[0]["mode_flags"]
                    },
                    1: {
                        "residual": frame_result[1]["residual"],
                        "mode_flags": frame_result[1]["mode_flags"]
                    },
                    2: {
                        "residual": frame_result[2]["residual"],
                        "mode_flags": frame_result[2]["mode_flags"]
                    }
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
        dct = bd(width=width, height=height, block_size=8)
        dct_uv = bd(width=width // 2, height=height // 2, block_size=8)

        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size

            # Get frame data
            frame_data = data[start_idx:end_idx]


            # Separate YUV components (returns [Y, U, V] as indices 0, 1, 2)
            components = separate_yuv_compression(frame_data, width, height)

            original_y = components[0].astype(np.float64)
            original_u = components[1].astype(np.float64)
            original_v = components[2].astype(np.float64)

            # Apply transformation strategy
            frame_result_y = dct.transform(original_y)
            frame_result_u = dct_uv.transform(original_u)
            frame_result_v = dct_uv.transform(original_v)

            # Store compressed frame data
            compressed_frames.append(
                {
                    "y_residual": frame_result_y,
                    "u_residual": frame_result_u,
                    "v_residual": frame_result_v,
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

    def _process_yuv_file_with_dct_and_quant(
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
        dct = bd(width=width, height=height, block_size=8)
        dct_uv = bd(width=width // 2, height=height // 2, block_size=8)
        quant = qs()

        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size

            # Get frame data
            frame_data = data[start_idx:end_idx]

            # Separate YUV components (returns [Y, U, V] as indices 0, 1, 2)
            components = separate_yuv_compression(frame_data, width, height)

            original_y = components[0].astype(np.float64)
            original_u = components[1].astype(np.float64)
            original_v = components[2].astype(np.float64)
            # Apply transformation strategy
            frame_result_y = dct.transform(original_y)
            frame_result_u = dct_uv.transform(original_u)
            frame_result_v = dct_uv.transform(original_v)
            # quantization
            quant_y, min_y, step_y = quant.process(frame_result_y, levels=128)
            quant_u, min_u, step_u = quant.process(frame_result_u, levels=128)
            quant_v, min_v, step_v = quant.process(frame_result_v, levels=128)
            # Store compressed frame data
            compressed_frames.append(
                {
                    "y_residual": quant_y,
                    "y_min": min_y,
                    "y_step": step_y,
                    "u_residual": quant_u,
                    "u_min": min_u,
                    "u_step": step_u,
                    "v_residual": quant_v,
                    "v_min": min_v,
                    "v_step": step_v,
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

    def decompress_video(self, compressed_path: str, output_path: str = None, borders: bool = False) -> str:
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

        decoder = DecodePredictionStrategy()

        # Open output file for writing
        with open(output_path, "wb") as f:
            # Process each frame
            for frame_data in compressed_data["frames"]:
                # Decode Y, U, and V components using indices 0, 1, 2
                y_decoded = decoder.decode(
                    frame_data[0]["residual"], frame_data[0]["mode_flags"], block_size=8, borders=borders
                )
                u_decoded = decoder.decode(
                    frame_data[1]["residual"], frame_data[1]["mode_flags"], block_size=8, borders=borders
                )
                v_decoded = decoder.decode(
                    frame_data[2]["residual"], frame_data[2]["mode_flags"], block_size=8, borders=borders
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

    def decompress_video_with_dct(
        self, compressed_path: str, output_path: str = None
    ) -> str:
        """
        Decompress a previously compressed video file with DCT (from compact binary format).

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

        # Load compressed data from binary
        with open(compressed_path, "rb") as f:
            # Read header
            width, height, num_frames = struct.unpack("iii", f.read(12))
            dct = bd(width=width, height=height, block_size=8)
            dct_uv = bd(width=width // 2, height=height // 2, block_size=8)
            frames = []
            for _ in range(num_frames):
                frame = {}
                for key, dct_obj in zip(
                    ["y_residual", "u_residual", "v_residual"], [dct, dct_uv, dct_uv]
                ):
                    # Read shape
                    rows, cols = struct.unpack("ii", f.read(8))
                    # Read data
                    arr_bytes = f.read(rows * cols * 4)  # float32
                    arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(
                        (rows, cols)
                    )
                    frame[key] = arr
                frames.append(frame)

        # Open output file for writing
        with open(output_path, "wb") as f_out:
            for frame_data in frames:
                y_decoded = dct.inverse_transform(frame_data["y_residual"])
                u_decoded = dct_uv.inverse_transform(frame_data["u_residual"])
                v_decoded = dct_uv.inverse_transform(frame_data["v_residual"])
                f_out.write(y_decoded.astype(np.uint8).tobytes())
                f_out.write(u_decoded.astype(np.uint8).tobytes())
                f_out.write(v_decoded.astype(np.uint8).tobytes())

        print(f"Decompressed video saved to {output_path}")
        return output_path

    def decompress_video_with_dct_and_quant(
        self, compressed_path: str, output_path: str = None
    ) -> str:
        """
        Decompress a previously compressed video file with DCT and quantization (from compact binary format).
        """
        output_dir = "test_videos/decompressed/"
        os.makedirs(output_dir, exist_ok=True)

        if output_path is None:
            filename = os.path.basename(compressed_path)
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.yuv"
            )

        quant = qs()

        with open(compressed_path, "rb") as f:
            # Leer header
            width, height, num_frames, block_size, levels = struct.unpack("iiiii", f.read(20))
            dct = bd(width=width, height=height, block_size=block_size)
            dct_uv = bd(width=width // 2, height=height // 2, block_size=block_size)

            # Determinar tipo de dato según levels
            if levels <= 256:
                dtype = np.uint8
                bytes_per_elem = 1
            elif levels <= 65536:
                dtype = np.uint16
                bytes_per_elem = 2
            elif levels <= 2**32:
                dtype = np.uint32
                bytes_per_elem = 4
            else:
                dtype = np.uint64
                bytes_per_elem = 8

            frames = []
            for _ in range(num_frames):
                frame = {}
                for key, dct_obj in zip(
                    ["y_residual", "u_residual", "v_residual"], [dct, dct_uv, dct_uv]
                ):
                    # Leer shape
                    rows, cols = struct.unpack("ii", f.read(8))
                    # Leer min y step
                    min_val, step = struct.unpack("ff", f.read(8))
                    # Leer datos cuantizados
                    arr_bytes = f.read(rows * cols * bytes_per_elem)
                    quant_arr = np.frombuffer(arr_bytes, dtype=dtype).reshape((rows, cols))
                    # Dequantizar
                    arr = quant.dequantize(quant_arr, min_val, step)
                    frame[key] = arr
                frames.append(frame)

        with open(output_path, "wb") as f_out:
            for frame_data in frames:
                y_decoded = dct.inverse_transform(frame_data["y_residual"])
                u_decoded = dct_uv.inverse_transform(frame_data["u_residual"])
                v_decoded = dct_uv.inverse_transform(frame_data["v_residual"])
                f_out.write(y_decoded.astype(np.uint8).tobytes())
                f_out.write(u_decoded.astype(np.uint8).tobytes())
                f_out.write(v_decoded.astype(np.uint8).tobytes())

        print(f"Decompressed video saved to {output_path}")
        return output_path
