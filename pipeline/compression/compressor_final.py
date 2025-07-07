import os
from numba import njit
import numpy as np

from pipeline.stages.bitwriter.bitwritter import BitWriter
from pipeline.stages.decorrelation.strategy.decode_prediction_strategy import _decode_block_float
from pipeline.stages.decorrelation.strategy.prediction_strategy import _to_mode_flag, find_best_mode_and_residuals_float, find_best_mode_and_residuals_uint8
from pipeline.stages.decorrelation.strategy.transformation_strategy import dct_block, idct_block, precompute_cosines
from pipeline.stages.quantization.quantization_stage import dequantize, quantize

class CompressorFinal:
    """
    Final version of the video compressor with all features integrated.
    """
    def __init__(self):
        pass


    def compress_video(self, video_path: str, output_path: str, height: int, width: int, block_size: int, levels: int):
        """
        Compresses a raw YUV video and saves it in binary format using BitWriter.
        """
        reader = BitWriter(video_path)
        # Read the original video as bytes
        original_video = reader.read_original_video()
        cosines = precompute_cosines(block_size)
        processed_video = self.process_video_compression(
            video_data=original_video,
            width=width,
            height=height,
            block_size=block_size,
            levels=levels,
            cosines=cosines
        )

        # Create output folder if it does not exist
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.basename(video_path)
        output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.finalcomp")

        # Write the compressed file using BitWriter
        writer = BitWriter(output_file)
        num_frames = len(processed_video)
        writer.write_compressed_video(processed_video, width, height, num_frames, block_size, levels)
        print(f"[compress_video] compressed file written in: {output_file}")

    def decompress_video(self, compressed_path: str, output_path: str):
        """
        Decompresses a binary compressed file (header + frames + block metadata)
        and reconstructs the raw YUV video.
        """
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(compressed_path)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.yuv")

        # Read the entire compressed file
        reader = BitWriter(compressed_path)
        compressed_data = reader.read_compressed_video()

        # Deserialize the compressed data
        frames, width, height, num_frames, block_size, levels = compressed_data
        cosines = precompute_cosines(block_size)
        decompressed_video = self.process_video_decompression(frames, width, height, num_frames, block_size, levels, cosines)

        # Write the reconstructed video
        writer = BitWriter(output_file)
        writer.write_reconstructed_video(decompressed_video)
        print(f"[decompress_video] Decompressed file written in: {output_file}")

    def process_video_compression(self, video_data: np.ndarray, width: int, height: int, block_size: int, levels: int, cosines: np.ndarray):
        """
        Receives a raw YUV420 video (np.ndarray 1D uint8), splits it into frames, separates channels, and applies compress_frame_test to each channel of each frame.
        Returns a list of compressed results per frame and channel: [[y_out, u_out, v_out], ...]
        """
        frame_size = width * height * 3 // 2
        num_frames = video_data.size // frame_size
        width_uv = width // 2
        height_uv = height // 2
        results = []
        for frame_idx in range(num_frames):
            frame_start = frame_idx * frame_size
            frame_end = frame_start + frame_size
            frame_bytes = video_data[frame_start:frame_end]
            # Separate YUV channels (assuming planar)
            y = frame_bytes[:width*height].reshape((height, width))
            u = frame_bytes[width*height:width*height + width_uv*height_uv].reshape((height_uv, width_uv))
            v = frame_bytes[width*height + width_uv*height_uv:].reshape((height_uv, width_uv))

            # Padding for each channel
            padded_w_y = ((width + block_size - 1) // block_size) * block_size
            padded_h_y = ((height + block_size - 1) // block_size) * block_size
            padded_w_uv = ((width_uv + block_size - 1) // block_size) * block_size
            padded_h_uv = ((height_uv + block_size - 1) // block_size) * block_size

            y_out = compress_frame_test(y.flatten(), width, height, padded_w_y, padded_h_y, block_size, levels, cosines)
            u_out = compress_frame_test(u.flatten(), width_uv, height_uv, padded_w_uv, padded_h_uv, block_size, levels, cosines)
            v_out = compress_frame_test(v.flatten(), width_uv, height_uv, padded_w_uv, padded_h_uv, block_size, levels, cosines)
            results.append([y_out, u_out, v_out])
            print(f"compressed frame {frame_idx + 1}/{num_frames}")
        return results

    def process_video_decompression(self, frames, width: int, height: int, num_frames: int, block_size: int, levels: int, cosines: np.ndarray):
        """
        Receives the deserialized structure (frames) and individual metadata,
        reconstructs the raw YUV video (np.ndarray 1D uint8).
        """
        width_uv = width // 2
        height_uv = height // 2
        frame_size = width * height + 2 * (width_uv * height_uv)
        output = np.empty(num_frames * frame_size, dtype=np.uint8)

        for frame_idx in range(num_frames):
            frame = frames[frame_idx]
            y_tuple, u_tuple, v_tuple = frame
            y_rec = decompress_frame_test(y_tuple, width, height, block_size, levels, cosines)
            u_rec = decompress_frame_test(u_tuple, width_uv, height_uv, block_size, levels, cosines)
            v_rec = decompress_frame_test(v_tuple, width_uv, height_uv, block_size, levels, cosines)
            start = frame_idx * frame_size
            output[start:start + width*height] = y_rec.flatten()
            output[start + width*height : start + width*height + width_uv*height_uv] = u_rec.flatten()
            output[start + width*height + width_uv*height_uv : start + frame_size] = v_rec.flatten()
            print(f"decompressed frame {frame_idx + 1}/{num_frames}")
        return output
@njit
def compress_frame_test(
    frame_data: bytes,
    width: int,
    height: int,
    padded_w: int,
    padded_h: int,
    block_size: int,
    levels: int,
    cosines: np.ndarray
):
    """
    Processes a frame using prediction + DCT + quantization compression.
    Updates the working frame with the reconstructed block after each block.
    """
    assert padded_w >= width, f"[compress_frame] padded_w ({padded_w}) < width ({width})"
    assert padded_h >= height, f"[compress_frame] padded_h ({padded_h}) < height ({height})"
    assert len(frame_data) == width * height, f"[compress_frame] frame_data size ({len(frame_data)}) does not match width*height ({width*height})"

    # 1. Convert frame data to numpy array
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = frame.reshape((height, width))

    # 2. Create padded frame
    padded_frame = np.zeros((padded_h, padded_w), dtype=np.uint8)
    padded_frame[:height, :width] = frame

    # 3. Calculate the number of blocks
    n_blocks_y = padded_h // block_size
    n_blocks_x = padded_w // block_size

    # 4. Preallocate arrays for the results
    processed_blocks = np.empty((n_blocks_y, n_blocks_x, block_size, block_size), dtype=np.uint8)
    mode_flags = np.empty((n_blocks_y, n_blocks_x), dtype=np.uint8)
    min_vals = np.empty((n_blocks_y, n_blocks_x), dtype=np.float32)
    steps = np.empty((n_blocks_y, n_blocks_x), dtype=np.float32)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y = by * block_size
            x = bx * block_size
            block = padded_frame[y:y+block_size, x:x+block_size]
            assert block.shape == (block_size, block_size), f"[compress_frame] Block shape mismatch at ({by},{bx}): {block.shape}"

            quantized_dct_residual, reconstructed_block, mode_flag, min_val, step = compress_block_prediction_dct_and_quant(
                block,
                i_start=0,
                j_start=0,
                block_size=block_size,
                levels=levels,
                cosines=cosines
            )
            # Store only the quantized block (as uint8 for storage)
            processed_blocks[by, bx] = quantized_dct_residual.astype(np.uint8)
            mode_flags[by, bx] = mode_flag
            min_vals[by, bx] = min_val
            steps[by, bx] = step

            # Replace the original block with the reconstructed one for future predictions
            for i in range(block_size):
                for j in range(block_size):
                    val = reconstructed_block[i, j]
                    if val < 0:
                        val = 0
                    elif val > 255:
                        val = 255
                    padded_frame[y + i, x + j] = int(val)

    # Returns tuples of arrays
    return processed_blocks, mode_flags, min_vals, steps

@njit
def decompress_frame_test(
    frame_data: tuple,
    width: int,
    height: int,
    block_size: int,
    levels: int,
    cosines: np.ndarray
) -> np.ndarray:
    """
    Decompresses a full frame from the compressed data (prediction + DCT + quantization)
    and returns only the reconstructed frame (np.ndarray uint8).
    """
    processed_blocks, mode_flags, min_vals, steps = frame_data

    padded_h = ((height + block_size - 1) // block_size) * block_size
    padded_w = ((width + block_size - 1) // block_size) * block_size

    # Preallocate the reconstructed frame
    reconstructed_frame = np.zeros((padded_h, padded_w), dtype=np.float64)

    n_blocks_y = padded_h // block_size
    n_blocks_x = padded_w // block_size

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y = by * block_size
            x = bx * block_size
            quantized_block = processed_blocks[by, bx]
            mode_flag = mode_flags[by, bx]
            min_val = min_vals[by, bx]
            step = steps[by, bx]

            decompressed_block = decompress_block_prediction_dct_and_quant(
                quantized_block,
                mode_flag,
                min_val,
                step,
                cosines,
                block_size
            )

            reconstructed_frame[y:y+block_size, x:x+block_size] = decompressed_block[:block_size, :block_size]

    # Crop the frame to its original size and return as uint8
    return reconstructed_frame[:height, :width].astype(np.uint8)

@njit
def compress_block_prediction_dct_and_quant(
    frame: np.ndarray,
    i_start: int,
    j_start: int,
    block_size: int,
    levels: int,
    cosines: np.ndarray
):
    """
    Applies prediction, then DCT to the residual, quantization, and block reconstruction.
    """
    # Extract the block
    block = frame[i_start:i_start+block_size, j_start:j_start+block_size]

    # 1. Prediction (no offset, float)
    best_mode, block_residuals = find_best_mode_and_residuals_float(
        block, 0, 0, block_size, block_size, True
    )
    mode_flag = _to_mode_flag(best_mode)

    # 2. DCT to the residual
    dct_residual = dct_block(block_residuals, cosines, block_size)

    # 3. Quantization of the DCT residual
    quantized_dct_residual, min_val, step = quantize(dct_residual, levels)

    # 4. Dequantization and IDCT for reconstruction
    dequantized_dct_residual = dequantize(quantized_dct_residual, min_val, step)
    residual_block_recon = idct_block(dequantized_dct_residual, cosines, block_size)

    # 5. Inverse prediction reconstruction
    reconstructed_block = np.zeros_like(residual_block_recon, dtype=np.float32)
    _decode_block_float(residual_block_recon, reconstructed_block, 0, 0, block_size, block_size, mode_flag, True)

    # Returns the quantized DCT residual, the reconstructed block, and metadata
    return quantized_dct_residual, reconstructed_block, mode_flag, min_val, step

@njit
def decompress_block_prediction_dct_and_quant(
    quantized_dct_residual: np.ndarray,
    mode_flag: int,
    min_val: float,
    step: float,
    cosines: np.ndarray,
    block_size: int
) -> np.ndarray:
    """
    Dequantizes, applies IDCT, and reconstructs the original block using prediction.
    """
    # 1. Dequantization of the DCT residual
    dequantized_dct_residual = dequantize(quantized_dct_residual, min_val, step)

    # 2. IDCT to obtain the spatial residual
    residual_block = idct_block(dequantized_dct_residual, cosines, block_size)

    # 3. Inverse prediction reconstruction
    reconstructed_block = np.zeros_like(residual_block, dtype=np.float32)
    _decode_block_float(residual_block, reconstructed_block, 0, 0, block_size, block_size, mode_flag, True)
    return reconstructed_block
