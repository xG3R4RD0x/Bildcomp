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
        Interfaz estándar para cuantizar (puede extenderse para batches o canales).
        Realiza la selección dinámica de tipo fuera de Numba.
        """
        quantized, min_val, step = self.quantize(array, levels)
        dtype = _select_dtype(levels)
        quantized = quantized.astype(dtype)
        return quantized, min_val, step
@njit
def quantize(array: np.ndarray, levels: int = 128):
        """
        Cuantiza un array numpy en el rango de valores usando 'levels' niveles.
        Devuelve el array cuantizado (uint32), el valor mínimo y el step.
        """
        arr = array.astype(np.float64)
        min_val = arr.min()
        max_val = arr.max()
        range_val = max_val - min_val
        if levels <= 1 or range_val == 0:
            # Evita división por cero
            return np.zeros_like(arr, dtype=np.uint32), min_val, 1.0
        step = range_val / levels
        quantized = np.floor((arr - min_val) / step)
        quantized = np.clip(quantized, 0, levels - 1)
        return quantized.astype(np.uint32), min_val, step


@njit
def dequantize(quantized: np.ndarray, min_val: float, step: float) -> np.ndarray:
        """
        Reconstruye el array aproximado a partir del cuantizado, el mínimo y el step.
        """
        return quantized.astype(np.float64) * step + min_val