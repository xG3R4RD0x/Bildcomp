import numpy as np
from numba import njit


class DecodePredictionStrategy:
    """
    Decodifica los residuales en uint8 y reconstruye el frame original usando los mode flags.
    Usa predicciones:
    - 0b00: Vertical
    - 0b01: Horizontal
    - 0b10: Promedio (Average)
    """

    def __init__(self):
        pass

    def decode(
        self, residuals: np.ndarray, mode_flags: np.ndarray, block_size: int = 8
    ) -> np.ndarray:
        height, width = residuals.shape
        reconstructed = np.zeros((height, width), dtype=np.uint8)

        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size

        for block_i in range(blocks_h):
            for block_j in range(blocks_w):
                i = block_i * block_size
                j = block_j * block_size
                block_h = min(block_size, height - i)
                block_w = min(block_size, width - j)

                mode_idx_i = min(block_i, mode_flags.shape[0] - 1)
                mode_idx_j = min(block_j, mode_flags.shape[1] - 1)
                mode_flag = mode_flags[mode_idx_i, mode_idx_j]

                if mode_flag == 0b00:  # Vertical
                    _decode_block(residuals, reconstructed, i, j, block_h, block_w, 0)
                elif mode_flag == 0b01:  # Horizontal
                    _decode_block(residuals, reconstructed, i, j, block_h, block_w, 1)
                elif mode_flag == 0b10:  # Average
                    _decode_block(residuals, reconstructed, i, j, block_h, block_w, 2)

        return reconstructed


@njit
def _decode_block(residuals, reconstructed, i, j, block_h, block_w, mode):
    for row in range(block_h):
        for col in range(block_w):
            abs_i = i + row
            abs_j = j + col

            # Predict value using the selected mode
            if mode == 0:  # Vertical
                if abs_i == 0:
                    pred = reconstructed[abs_i, abs_j - 1] if abs_j > 0 else 128
                else:
                    pred = reconstructed[abs_i - 1, abs_j]
            elif mode == 1:  # Horizontal
                if abs_j == 0:
                    pred = reconstructed[abs_i - 1, abs_j] if abs_i > 0 else 128
                else:
                    pred = reconstructed[abs_i, abs_j - 1]
            else:  # Average
                if abs_i == 0 and abs_j == 0:
                    pred = 128
                elif abs_i == 0:
                    pred = reconstructed[abs_i, abs_j - 1]
                elif abs_j == 0:
                    pred = reconstructed[abs_i - 1, abs_j]
                else:
                    pred = (
                        reconstructed[abs_i - 1, abs_j]
                        + reconstructed[abs_i, abs_j - 1]
                    ) // 2

            # Decode residual: residual = 128 + error → error = residual - 128
            res = residuals[abs_i, abs_j]
            error = int(res) - 128
            reconstructed[abs_i, abs_j] = (int(pred) + error) % 256
