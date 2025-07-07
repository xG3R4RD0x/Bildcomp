import numpy as np
from numba import njit
from pipeline.interfaces.base_stage import CompressionStage


def _select_dtype(levels):
    if levels <= 256:
        return np.uint8
    elif levels <= 65536:
        return np.uint16
    elif levels <= 2**32:
        return np.uint32
    else:
        return np.uint64


class QuantizationStage(CompressionStage):
    def name(self) -> str:
        return "Quantization Stage"

    def process(self, array: np.ndarray, levels: int = 128):
        """
        Standard interface for quantization (can be extended for batches or channels).
        Performs dynamic type selection outside of Numba.
        """
        quantized, min_val, step = self.quantize(array, levels)
        dtype = _select_dtype(levels)
        quantized = quantized.astype(dtype)
        return quantized, min_val, step
@njit
def quantize(array: np.ndarray, levels: int = 128):
    """
    Quantizes a numpy array in the value range using 'levels' levels.
    Returns the quantized array (uint32), the minimum value, and the step.
    """
    arr = array.astype(np.float64)
    min_val = arr.min()
    max_val = arr.max()
    range_val = max_val - min_val
    if levels <= 1 or range_val == 0:
        # Avoid division by zero
        return np.zeros_like(arr, dtype=np.uint32), min_val, 1.0
    step = range_val / levels
    quantized = np.floor((arr - min_val) / step)
    quantized = np.clip(quantized, 0, levels - 1)
    return quantized.astype(np.uint32), min_val, step


@njit
def dequantize(quantized: np.ndarray, min_val: float, step: float) -> np.ndarray:
    """
    Reconstructs the approximate array from the quantized data, the minimum value, and the step.
    """
    return quantized.astype(np.float64) * step + min_val