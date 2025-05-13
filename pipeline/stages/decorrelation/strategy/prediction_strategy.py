import numpy as np
from typing import Dict, Any
from numba import njit


class PredictionStrategy:
    def __init__(self):
        self.mode_flags = {"vertical": 0b00, "horizontal": 0b01, "average": 0b10}

    def process(self, data: Dict[str, Any], block_size: int = 8) -> Dict[str, Any]:
        results = {}
        for component in ["y", "u", "v"]:
            component_data = np.copy(data[component])
            height, width = component_data.shape
            blocks_h = (height + block_size - 1) // block_size
            blocks_w = (width + block_size - 1) // block_size

            residuals = np.zeros((height, width), dtype=np.uint8)
            mode_flags_map = np.zeros((blocks_h, blocks_w), dtype=np.uint8)

            for bi in range(blocks_h):
                for bj in range(blocks_w):
                    i_start = bi * block_size
                    j_start = bj * block_size
                    i_end = min(i_start + block_size, height)
                    j_end = min(j_start + block_size, width)

                    best_mode, block_residuals = find_best_mode_and_residuals_uint8(
                        component_data, i_start, j_start, i_end, j_end
                    )

                    mode_flags_map[bi, bj] = self.mode_flags[best_mode]
                    residuals[i_start:i_end, j_start:j_end] = block_residuals

            results[f"{component}_residual"] = residuals
            results[f"{component}_mode_flags"] = mode_flags_map

        return results


@njit(inline="always")
def predict_vertical(data, i, j):
    return data[i - 1, j] if i > 0 else (data[i, j - 1] if j > 0 else 128)


@njit(inline="always")
def predict_horizontal(data, i, j):
    return data[i, j - 1] if j > 0 else (data[i - 1, j] if i > 0 else 128)


@njit(inline="always")
def predict_average(data, i, j):
    if i == 0 and j == 0:
        return 128
    elif i == 0:
        return data[i, j - 1]
    elif j == 0:
        return data[i - 1, j]
    else:
        return (data[i - 1, j] + data[i, j - 1]) // 2


@njit
def apply_prediction_block_uint8(data, i_start, j_start, i_end, j_end, mode):
    h = i_end - i_start
    w = j_end - j_start
    residuals = np.zeros((h, w), dtype=np.uint8)

    for i_off in range(h):
        for j_off in range(w):
            i = i_start + i_off
            j = j_start + j_off

            if mode == 0:
                pred = predict_vertical(data, i, j)
            elif mode == 1:
                pred = predict_horizontal(data, i, j)
            else:
                pred = predict_average(data, i, j)

            actual = data[i, j]
            # Residual en rango [0, 255] como en Octave:
            diff = int(actual) - int(pred)
            if diff < -128:
                diff = 128 + diff + 256
            elif diff > 127:
                diff = 128 + diff - 256
            else:
                diff = 128 + diff
            residuals[i_off, j_off] = diff

    return residuals


@njit
def find_best_mode_and_residuals_uint8(data, i_start, j_start, i_end, j_end):
    res_v = apply_prediction_block_uint8(data, i_start, j_start, i_end, j_end, 0)
    res_h = apply_prediction_block_uint8(data, i_start, j_start, i_end, j_end, 1)
    res_a = apply_prediction_block_uint8(data, i_start, j_start, i_end, j_end, 2)

    err_v = np.sum(np.abs(res_v - 128))
    err_h = np.sum(np.abs(res_h - 128))
    err_a = np.sum(np.abs(res_a - 128))

    if err_v <= err_h and err_v <= err_a:
        return "vertical", res_v
    elif err_h <= err_a:
        return "horizontal", res_h
    else:
        return "average", res_a
