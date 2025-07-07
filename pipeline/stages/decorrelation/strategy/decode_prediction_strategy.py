import numpy as np
from numba import njit


# --- Copia las funciones de predicción del encoder para asegurar simetría ---
@njit(inline="always")
def predict_vertical(data, i, j, borders=False, start_i=0, start_j=0):
    if not borders:
        return data[i - 1, j] if i > 0 else (data[i, j - 1] if j > 0 else 128)
    else:
        if i > start_i:
            return data[i - 1 - start_i, j]
        elif j > start_j:
            return data[i, j - 1 - start_j]
        else:
            return 128


@njit(inline="always")
def predict_horizontal(data, i, j, borders=False, start_i=0, start_j=0):
    if not borders:
        return data[i, j - 1] if j > 0 else (data[i - 1, j] if i > 0 else 128)
    else:
        if j > start_j:
            return data[i, j - 1 - start_j]
        elif i > start_i:
            return data[i - 1 - start_i, j]
        else:
            return 128


@njit(inline="always")
def predict_average(data, i, j, borders=False, start_i=0, start_j=0):
    if i == 0 and j == 0:
        return 128
    elif not borders:
        if i == 0:
            return data[i, j - 1]
        elif j == 0:
            return data[i - 1, j]
        else:
            return (data[i - 1, j] + data[i, j - 1]) // 2
    else:
        if i == start_i and j == start_j:
            return 128
        elif i == start_i:
            return data[i, j - 1 - start_j]
        elif j == start_j:
            return data[i - 1 - start_i, j]
        else:
            return (data[i - 1 - start_i, j] + data[i, j - 1 - start_j]) // 2


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
        self, residuals: np.ndarray, mode_flags: np.ndarray, block_size: int = 8, borders: bool = False
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

                _decode_block(
                    residuals, reconstructed, i, j, block_h, block_w, mode_flag, borders
                )

        return reconstructed


@njit
def _decode_block(residuals, reconstructed, i, j, block_h, block_w, mode_flag, borders):
    for row in range(block_h):
        for col in range(block_w):
            abs_i = i + row
            abs_j = j + col

            if mode_flag == 0b00:
                pred = predict_vertical(reconstructed, abs_i, abs_j, borders, i, j)
            elif mode_flag == 0b01:
                pred = predict_horizontal(reconstructed, abs_i, abs_j, borders, i, j)
            else:
                pred = predict_average(reconstructed, abs_i, abs_j, borders, i, j)

            res = residuals[abs_i, abs_j]
            error = int(res) - 128
            reconstructed[abs_i, abs_j] = (int(pred) + error) % 256
            
@njit
def _decode_block_float(residuals, reconstructed, i, j, block_h, block_w, mode_flag, borders):
    for row in range(block_h):
        for col in range(block_w):
            abs_i = i + row
            abs_j = j + col

            if mode_flag == 0b00:
                pred = predict_vertical(reconstructed, abs_i, abs_j, borders, i, j)
            elif mode_flag == 0b01:
                pred = predict_horizontal(reconstructed, abs_i, abs_j, borders, i, j)
            else:
                pred = predict_average(reconstructed, abs_i, abs_j, borders, i, j)

            res = residuals[abs_i, abs_j]
            # Para float32 centrado en 0, no hay offset
            val = int(round(pred + res))
            if val < 0:
                val = 0
            elif val > 255:
                val = 255
            reconstructed[abs_i, abs_j] = val
