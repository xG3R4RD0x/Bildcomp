import os
import struct
from numba import njit
import numpy as np

from pipeline.stages.bitwriter.bitwritter import BitWriter
from pipeline.stages.decorrelation.strategy.decode_prediction_strategy import _decode_block, _decode_block_float
from pipeline.stages.decorrelation.strategy.prediction_strategy import _to_mode_flag, find_best_mode_and_residuals_float, find_best_mode_and_residuals_uint8
from pipeline.stages.decorrelation.strategy.transformation_strategy import dct_block, idct_block, precompute_cosines
from pipeline.stages.quantization.quantization_stage import dequantize, quantize
from pipeline.stages.decorrelation.decorrelation_stage import separate_yuv_compression

class CompressorFinal:
    """
    Final version of the video compressor with all features integrated.
    """
    def __init__(self):
        pass


    def compress_video(self, video_path: str, output_path: str, height: int, width: int, block_size: int, levels: int):
        """
        Comprime un video YUV plano y lo guarda en formato binario usando BitWriter.
        """
        reader = BitWriter(video_path)
        # Leer el video original como bytes
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

        # Crear carpeta de salida si no existe
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.basename(video_path)
        output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.finalcomp")

        # Escribir el archivo comprimido usando BitWriter
        writer = BitWriter(output_file)
        num_frames = len(processed_video)
        writer.write_compressed_video(processed_video, width, height, num_frames, block_size, levels)
        print(f"[compress_video] compressed file written in: {output_file}")

    def decompress_video(self, compressed_path: str, output_path: str):
        """
        Descomprime un archivo comprimido en formato binario (header + frames + metadatos por bloque)
        y reconstruye el video YUV plano.
        """
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(compressed_path)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.yuv")

        # Leer todo el archivo comprimido
        reader = BitWriter(compressed_path)
        compressed_data = reader.read_compressed_video()

        # Deserializar los datos comprimidos
        frames, width, height, num_frames, block_size, levels = compressed_data
        cosines = precompute_cosines(block_size)
        decompressed_video = self.process_video_decompression(frames, width, height, num_frames, block_size, levels, cosines)

        # Escribir el video reconstruido
        writer = BitWriter(output_file)
        writer.write_reconstructed_video(decompressed_video)
        print(f"[decompress_video] Decompressed file written in: {output_file}")

    def process_video_compression(self, video_data: np.ndarray, width: int, height: int, block_size: int, levels: int, cosines: np.ndarray):
        """
        Recibe un video YUV420 plano (np.ndarray 1D uint8), lo divide en frames, separa canales y aplica compress_frame_test a cada canal de cada frame.
        Devuelve una lista de resultados comprimidos por frame y canal: [[y_out, u_out, v_out], ...]
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
            # Separar canales YUV (asumiendo planar)
            y = frame_bytes[:width*height].reshape((height, width))
            u = frame_bytes[width*height:width*height + width_uv*height_uv].reshape((height_uv, width_uv))
            v = frame_bytes[width*height + width_uv*height_uv:].reshape((height_uv, width_uv))

            # Padding para cada canal
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
        Recibe la estructura deserializada (frames) y los metadatos individuales,
        reconstruye el video YUV plano (np.ndarray 1D uint8).
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
    def serialize_compressed_video(self, compressed_frames, width, height, num_frames, block_size, levels):
        """
        Serializa la estructura de salida de process_video_compression a un objeto bytes,
        usando el mismo formato que BitWriter.write_compressed_video.
        """
        import struct
        byte_chunks = []
        # Header: width, height, num_frames, block_size, levels (todos <HHHBB>)
        byte_chunks.append(struct.pack('<HHHBB', width, height, num_frames, block_size, levels))
        for frame in compressed_frames:
            for channel_tuple in frame:  # [y_out, u_out, v_out]
                processed_blocks, mode_flags, min_vals, steps = channel_tuple
                n_blocks_y, n_blocks_x = mode_flags.shape
                # Guarda shape (2 int32)
                byte_chunks.append(struct.pack("ii", n_blocks_y, n_blocks_x))
                # Guarda min_vals y steps (float32 por bloque)
                byte_chunks.append(min_vals.astype(np.float32).tobytes())
                byte_chunks.append(steps.astype(np.float32).tobytes())
                # Guarda todos los bloques (uint8)
                byte_chunks.append(processed_blocks.astype(np.uint8).tobytes())
                # Guarda todos los mode_flags (uint8)
                byte_chunks.append(mode_flags.astype(np.uint8).tobytes())
        return b"".join(byte_chunks)

    def deserialize_compressed_video(self, data: bytes):
        """
        Deserializa un archivo binario comprimido (como el generado por serialize_compressed_video o BitWriter)
        y reconstruye la estructura: [ [ (processed_blocks, mode_flags, min_vals, steps), ... ], ... ]
        Devuelve: (frames, width, height, num_frames, block_size, levels)
        """
        import struct
        import numpy as np

        offset = 0
        # Header: <HHHBB
        width, height, num_frames, block_size, levels = struct.unpack_from('<HHHBB', data, offset)
        offset += struct.calcsize('<HHHBB')
        frames = []
        for _ in range(num_frames):
            frame = []
            for _ in range(3):  # Y, U, V
                # Shape
                n_blocks_y, n_blocks_x = struct.unpack_from("ii", data, offset)
                offset += 8
                # min_vals
                min_vals = np.frombuffer(data, dtype=np.float32, count=n_blocks_y * n_blocks_x, offset=offset)
                min_vals = min_vals.reshape((n_blocks_y, n_blocks_x))
                offset += 4 * n_blocks_y * n_blocks_x
                # steps
                steps = np.frombuffer(data, dtype=np.float32, count=n_blocks_y * n_blocks_x, offset=offset)
                steps = steps.reshape((n_blocks_y, n_blocks_x))
                offset += 4 * n_blocks_y * n_blocks_x
                # processed_blocks
                block_size_sq = block_size * block_size
                num_blocks = n_blocks_y * n_blocks_x * block_size_sq
                processed_blocks = np.frombuffer(data, dtype=np.uint8, count=num_blocks, offset=offset)
                processed_blocks = processed_blocks.reshape((n_blocks_y, n_blocks_x, block_size, block_size))
                offset += num_blocks
                # mode_flags
                mode_flags = np.frombuffer(data, dtype=np.uint8, count=n_blocks_y * n_blocks_x, offset=offset)
                mode_flags = mode_flags.reshape((n_blocks_y, n_blocks_x))
                offset += n_blocks_y * n_blocks_x
                frame.append((processed_blocks, mode_flags, min_vals, steps))
            frames.append(frame)
        return frames, width, height, num_frames, block_size, levels


@njit
def compress_frame(frame_data: bytes, width: int, height: int, padded_w: int, padded_h: int, block_size: int, levels: int, cosines: np.ndarray):
    # print(f"[compress_frame] width={width}, height={height}, padded_w={padded_w}, padded_h={padded_h}, block_size={block_size}")
    # print(f"[compress_frame] frame_data size={len(frame_data)} expected={width*height}")
    assert padded_w >= width, f"[compress_frame] padded_w ({padded_w}) < width ({width})"
    assert padded_h >= height, f"[compress_frame] padded_h ({padded_h}) < height ({height})"
    assert len(frame_data) == width * height, f"[compress_frame] frame_data size ({len(frame_data)}) does not match width*height ({width*height})"

    # 1. Convert frame data to numpy array
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = frame.reshape((height, width))

    # 3. Create padded frame
    padded_frame = np.zeros((padded_h, padded_w), dtype=np.uint8)
    padded_frame[:height, :width] = frame

    # Calcula el número de bloques
    n_blocks_y = padded_h // block_size
    n_blocks_x = padded_w // block_size

    # Prealoca arrays para los resultados
    processed_blocks = np.empty((n_blocks_y, n_blocks_x, block_size, block_size), dtype=np.uint8)
    mode_flags = np.empty((n_blocks_y, n_blocks_x), dtype=np.uint8)
    min_vals = np.empty((n_blocks_y, n_blocks_x), dtype=np.float32)
    steps = np.empty((n_blocks_y, n_blocks_x), dtype=np.float32)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y = by * block_size
            x = bx * block_size
            block = padded_frame[y:y+block_size, x:x+block_size]
            # print(f"[compress_frame] Block ({by},{bx}) at y={y}, x={x}, block.shape={block.shape}")
            assert block.shape == (block_size, block_size), f"[compress_frame] Block shape mismatch at ({by},{bx}): {block.shape}"

            processed_block, reconstructed_block, mode_flag, min_val, step = compress_block_prediction_dct_and_quant(
                block,
                i_start=0,
                j_start=0,
                block_size=block_size,
                levels=levels,
                cosines=cosines
            )
            padded_frame[y:y+block_size, x:x+block_size] = reconstructed_block[:block.shape[0], :block.shape[1]]
            processed_blocks[by, bx] = processed_block.astype(np.uint8)
            mode_flags[by, bx] = mode_flag
            min_vals[by, bx] = min_val
            steps[by, bx] = step

    # Devuelve tuplas de arrays
    return processed_blocks, mode_flags, min_vals, steps

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
    Procesa un frame usando compresión por predicción + DCT + cuantización.
    Actualiza el frame de trabajo con el bloque reconstruido tras cada bloque.
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

    # 3. Calcula el número de bloques
    n_blocks_y = padded_h // block_size
    n_blocks_x = padded_w // block_size

    # 4. Prealoca arrays para los resultados
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
            # Guarda solo el bloque cuantizado (como uint8 para almacenamiento)
            processed_blocks[by, bx] = quantized_dct_residual.astype(np.uint8)
            mode_flags[by, bx] = mode_flag
            min_vals[by, bx] = min_val
            steps[by, bx] = step

            # Reemplaza el bloque original con el reconstruido para futuras predicciones
            for i in range(block_size):
                for j in range(block_size):
                    val = reconstructed_block[i, j]
                    if val < 0:
                        val = 0
                    elif val > 255:
                        val = 255
                    padded_frame[y + i, x + j] = int(val)

    # Devuelve tuplas de arrays
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
    Descomprime un frame completo a partir de los datos comprimidos (predicción + DCT + cuantización)
    y devuelve solo el frame reconstruido (np.ndarray uint8).
    """
    processed_blocks, mode_flags, min_vals, steps = frame_data

    padded_h = ((height + block_size - 1) // block_size) * block_size
    padded_w = ((width + block_size - 1) // block_size) * block_size

    # Prealoca el frame reconstruido
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

    # Recorta el frame a su tamaño original y lo devuelve como uint8
    return reconstructed_frame[:height, :width].astype(np.uint8)


@njit
def compress_block(frame: np.ndarray,
              
                   i_start: int,
                   j_start: int,
                   block_size: int,
                   levels: int,
                   cosines: np.ndarray):
    
    # 1. Limitar el bloque dentro de los límites del frame (ahora siempre es el bloque completo)
    i_end = block_size
    j_end = block_size

    # 2. Predicción (ya importaste la función compatible con njit)
    best_mode, block_residuals = find_best_mode_and_residuals_uint8(
        frame, i_start, j_start, i_end, j_end, True
    )

    # Convertir el modo a flag (asumimos que ya importaste una función njit-friendly)
    mode_flag = _to_mode_flag(best_mode)

    # 3. Transformada DCT
    transformed_block = dct_block(block_residuals, cosines, block_size)

    # 4. Cuantización
    quantized_block, min_val, step = quantize(transformed_block, levels)

    # 5. Descuantización
    dequantized_block = dequantize(quantized_block, min_val, step)

    # 6. Transformada inversa
    untransformed_block = idct_block(dequantized_block, cosines, block_size)

    # 7. Reconstrucción con estrategia de predicción
    reconstructed_block = np.zeros_like(untransformed_block)

    _decode_block(untransformed_block,
                  reconstructed_block,
                  i_start,
                  j_start,
                  untransformed_block.shape[0],
                  untransformed_block.shape[1],
                  mode_flag,
                  True)

    # 8. Retorno compatible con njit: tupla fija
    return quantized_block, reconstructed_block, mode_flag, min_val, step
@njit
def compress_block_prediction_float(
    frame: np.ndarray,
    i_start: int,
    j_start: int,
    block_size: int,
    levels: int,
    cosines: np.ndarray
):
    """
    Predicción en float, DCT, cuantización, y reconstrucción en float.
    """
    # 1. Predicción en float (sin offset)
    best_mode, block_residuals = find_best_mode_and_residuals_float(
        frame, i_start, j_start, i_start + block_size, j_start + block_size, True
    )
    mode_flag = _to_mode_flag(best_mode)

    # 2. DCT sobre el residual
    transformed_block = dct_block(block_residuals, cosines, block_size)

    # 3. Cuantización
    quantized_block, min_val, step = quantize(transformed_block, levels)

    # 4. Descuantización
    dequantized_block = dequantize(quantized_block, min_val, step)

    # 5. IDCT
    untransformed_block = idct_block(dequantized_block, cosines, block_size)

    # 6. Reconstrucción predictiva en float
    reconstructed_block = np.zeros_like(untransformed_block, dtype=np.float32)
    _decode_block_float(
        untransformed_block,
        reconstructed_block,
        0,
        0,
        block_size,
        block_size,
        mode_flag,
        True
    )

    return quantized_block, reconstructed_block, mode_flag, min_val, step

@njit
def decompress_block_prediction_float(
    quantized_block: np.ndarray,
    mode_flag: int,
    min_val: float,
    step: float,
    cosines: np.ndarray,
    block_size: int
) -> np.ndarray:
    """
    Descomprime un bloque cuantizado (predicción en float) y reconstruye el bloque original.
    """
    # 1. Descuantización
    dequantized_block = dequantize(quantized_block, min_val, step)

    # 2. IDCT
    untransformed_block = idct_block(dequantized_block, cosines, block_size)

    # 3. Reconstrucción predictiva en float
    reconstructed_block = np.zeros_like(untransformed_block, dtype=np.float32)
    _decode_block_float(
        untransformed_block,
        reconstructed_block,
        0,
        0,
        block_size,
        block_size,
        mode_flag,
        True
    )

    return reconstructed_block

@njit
def decompress_block(
    quantized_block: np.ndarray,
    mode_flag: int,
    min_val: float,
    step: float,
    cosines: np.ndarray,
    block_size: int
) -> np.ndarray:
    """
    Descomprime un bloque cuantizado y reconstruye el bloque original.
    """
    # 1. Descuantización
    dequantized_block = dequantize(quantized_block, min_val, step)

    # 2. Transformada inversa DCT
    untransformed_block = idct_block(dequantized_block, cosines, block_size)

    # 3. Reconstrucción con estrategia de predicción
    reconstructed_block = np.zeros_like(untransformed_block)
    
    _decode_block(untransformed_block, reconstructed_block, 0, 0, block_size, block_size, mode_flag, True)

    return reconstructed_block

@njit
def decompress_frame(
    frame_data: tuple,
    width: int,
    height: int,
    block_size: int,
    levels: int
) -> np.ndarray:
    """
    Descomprime un frame completo a partir de los datos comprimidos y devuelve solo el frame reconstruido (np.ndarray uint8).
    """
    processed_blocks, mode_flags, min_vals, steps = frame_data

    padded_h = ((height + block_size - 1) // block_size) * block_size
    padded_w = ((width + block_size - 1) // block_size) * block_size

    cosines = precompute_cosines(block_size)

    # Prealoca el frame reconstruido
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

    # Recorta el frame a su tamaño original y lo devuelve como uint8
    return reconstructed_frame[:height, :width].astype(np.uint8)

@njit
def compress_block_transformation(frame: np.ndarray,
                   i_start: int,
                   j_start: int,
                   block_size: int,
                   levels: int,
                   cosines: np.ndarray):
    """
    Solo aplica la DCT y la inversa, sin predicción ni cuantización.
    """
    # Extrae el bloque
    block = frame[i_start:i_start+block_size, j_start:j_start+block_size]

    # 1. Transformada DCT
    transformed_block = dct_block(block, cosines, block_size)


    # Retorna el bloque transformado y el reconstruido
    return transformed_block

@njit
def decompress_block_transformation(
    transformed_block: np.ndarray,
    cosines: np.ndarray,
    block_size: int
) -> np.ndarray:
    """
    Solo aplica la transformada inversa DCT.
    """
    # 1. Transformada inversa DCT
    untransformed_block = idct_block(transformed_block, cosines, block_size)
    return untransformed_block
@njit
def compress_block_transformation_and_quant(frame: np.ndarray,
                   i_start: int,
                   j_start: int,
                   block_size: int,
                   levels: int,
                   cosines: np.ndarray):
    """
    Solo aplica la DCT y la inversa, sin predicción ni cuantización.
    """
    # Extrae el bloque
    block = frame[i_start:i_start+block_size, j_start:j_start+block_size]

    # 1. Transformada DCT
    transformed_block = dct_block(block, cosines, block_size)
     # 2. Cuantización
    quantized_block, min_val, step = quantize(transformed_block, levels)


    # Retorna el bloque transformado y el reconstruido
    return quantized_block, min_val, step

@njit
def decompress_block_transformation_and_quant(
    transformed_block: np.ndarray,
    min_val: float,
    step: float,
    cosines: np.ndarray,
    block_size: int
) -> np.ndarray:
    """
    Solo aplica la transformada inversa DCT.
    """
    # 1. Descuantización
    dequantized_block = dequantize(transformed_block, min_val, step)

    # 2. Transformada inversa DCT
    untransformed_block = idct_block(dequantized_block, cosines, block_size)
    return untransformed_block

@njit
def compress_block_prediction_and_quant(frame: np.ndarray,
                   i_start: int,
                   j_start: int,
                   block_size: int,
                   levels: int,
                   cosines: np.ndarray):
    """
    Solo aplica la predicción (sin DCT ni cuantización).
    """
    # Extrae el bloque
    block = frame[i_start:i_start+block_size, j_start:j_start+block_size]

    # 1. Predicción (usa find_best_mode_and_residuals_float)
    best_mode, block_residuals = find_best_mode_and_residuals_float(
        block, 0, 0, block_size, block_size, True
    )
    
    mode_flag = _to_mode_flag(best_mode)

    # 2. Cuantización del residual
    quantized_residual, min_val, step = quantize(block_residuals, levels)


    # Retorna el residual y el modo
    return quantized_residual, mode_flag, min_val, step

@njit
def decompress_block_prediction_and_quant(
    residual_block: np.ndarray,
    mode_flag: int,
    min_val: float,
    step: float,
    block_size: int
) -> np.ndarray:
    """
    Solo aplica la reconstrucción inversa de la predicción.
    """
     # 1. Descuantización del residual
    dequantized_residual = dequantize(residual_block, min_val, step)
    
    # 2. Reconstrucción inversa de la predicción
    
    reconstructed_block = np.zeros_like(residual_block, dtype=np.float32)
    _decode_block_float(dequantized_residual, reconstructed_block, 0, 0, block_size, block_size, mode_flag, True)
    return reconstructed_block
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
    Aplica predicción, luego DCT al residual, cuantización y reconstrucción del bloque.
    """
    # Extrae el bloque
    block = frame[i_start:i_start+block_size, j_start:j_start+block_size]

    # 1. Predicción (sin offset, float)
    best_mode, block_residuals = find_best_mode_and_residuals_float(
        block, 0, 0, block_size, block_size, True
    )
    mode_flag = _to_mode_flag(best_mode)

    # 2. DCT al residual
    dct_residual = dct_block(block_residuals, cosines, block_size)

    # 3. Cuantización del residual DCT
    quantized_dct_residual, min_val, step = quantize(dct_residual, levels)

    # 4. Descuantización e IDCT para reconstrucción
    dequantized_dct_residual = dequantize(quantized_dct_residual, min_val, step)
    residual_block_recon = idct_block(dequantized_dct_residual, cosines, block_size)

    # 5. Reconstrucción inversa de la predicción
    reconstructed_block = np.zeros_like(residual_block_recon, dtype=np.float32)
    _decode_block_float(residual_block_recon, reconstructed_block, 0, 0, block_size, block_size, mode_flag, True)

    # Retorna el residual DCT cuantizado, el bloque reconstruido y los metadatos
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
    Descuantiza, aplica IDCT y reconstruye el bloque original usando la predicción.
    """
    # 1. Descuantización del residual DCT
    dequantized_dct_residual = dequantize(quantized_dct_residual, min_val, step)

    # 2. IDCT para obtener el residual espacial
    residual_block = idct_block(dequantized_dct_residual, cosines, block_size)

    # 3. Reconstrucción inversa de la predicción
    reconstructed_block = np.zeros_like(residual_block, dtype=np.float32)
    _decode_block_float(residual_block, reconstructed_block, 0, 0, block_size, block_size, mode_flag, True)
    return reconstructed_block

@njit
def compress_block_prediction(frame: np.ndarray,
                   i_start: int,
                   j_start: int,
                   block_size: int,
                   levels: int,
                   cosines: np.ndarray):
    """
    Solo aplica la predicción (sin DCT ni cuantización).
    """
    # Extrae el bloque
    block = frame[i_start:i_start+block_size, j_start:j_start+block_size]

    # 1. Predicción (usa find_best_mode_and_residuals_uint8)
    best_mode, block_residuals = find_best_mode_and_residuals_uint8(
        block, 0, 0, block_size, block_size, True
    )
    mode_flag = _to_mode_flag(best_mode)

    # Retorna el residual y el modo
    return block_residuals, mode_flag

@njit
def decompress_block_prediction(
    residual_block: np.ndarray,
    mode_flag: int,
    block_size: int
) -> np.ndarray:
    """
    Solo aplica la reconstrucción inversa de la predicción.
    """
    reconstructed_block = np.zeros_like(residual_block)
    _decode_block(residual_block, reconstructed_block, 0, 0, block_size, block_size, mode_flag, True)
    return reconstructed_block