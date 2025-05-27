import numpy as np
from numba import njit

@njit
def precompute_cosines(N):
    cosines = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            cosines[k, n] = np.cos((2 * n + 1) * k * np.pi / (2 * N))
    return cosines

@njit
def dct_block(block, cosines, block_size):
    result = np.zeros((block_size, block_size), dtype=np.float64)
    for u in range(block_size):
        for v in range(block_size):
            Cu = 1 / np.sqrt(2) if u == 0 else 1
            Cv = 1 / np.sqrt(2) if v == 0 else 1
            sum_val = 0.0
            for x in range(block_size):
                for y in range(block_size):
                    sum_val += block[x, y] * cosines[u, x] * cosines[v, y]
            result[u, v] = 0.25 * Cu * Cv * sum_val
    return result

@njit
def idct_block(block, cosines, block_size):
    result = np.zeros((block_size, block_size), dtype=np.float64)
    for x in range(block_size):
        for y in range(block_size):
            sum_val = 0.0
            for u in range(block_size):
                for v in range(block_size):
                    Cu = 1 / np.sqrt(2) if u == 0 else 1
                    Cv = 1 / np.sqrt(2) if v == 0 else 1
                    sum_val += Cu * Cv * block[u, v] * cosines[u, x] * cosines[v, y]
            result[x, y] = 0.25 * sum_val
    return result

@njit
def process_blocks(frame, height, width, block_size, cosines, inverse=False):
    padded_h = ((height + block_size - 1) // block_size) * block_size
    padded_w = ((width + block_size - 1) // block_size) * block_size
    padded = np.zeros((padded_h, padded_w), dtype=np.float64)
    padded[:height, :width] = frame

    result = np.zeros_like(padded)
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            if inverse:
                result[i:i + block_size, j:j + block_size] = idct_block(block, cosines, block_size)
            else:
                result[i:i + block_size, j:j + block_size] = dct_block(block, cosines, block_size)
    return result[:height, :width]

class BlockDCT:
    def __init__(self, width, height, block_size=8):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.cosines = precompute_cosines(block_size)

    def transform(self, frame: np.ndarray) -> np.ndarray:
        return process_blocks(frame, self.height, self.width, self.block_size, self.cosines, inverse=False)

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        result = process_blocks(transformed, self.height, self.width, self.block_size, self.cosines, inverse=True)
        #tengo un margen de error en la transformación inversa
        #todos los pixeles tiene un valor -1 de lo que deberían ser
        #esto es porque la DCT no es exacta, y la IDCT tampoco
        #por lo tanto, redondeo el resultado para que sea un entero
        #y lo devuelvo como uint8
        return np.round(result)  
