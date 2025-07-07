import struct
import numpy as np

class BitWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def write_header(self, width, height, num_frames, block_size, levels):
        """
        Escribe el header del archivo comprimido.
        Header: [width (H)][height (H)][num_frames (H)][block_size (B)][levels (B)]
        """
        with open(self.file_path, 'wb') as f:
            f.write(struct.pack('<HHHBB', width, height, num_frames, block_size, levels))
                    
    def write_compressed_video(self, video_data, width, height, num_frames, block_size, levels):
        """
        Escribe los frames procesados (arrays y metadatos) en el archivo.
        """
        data_bytes = self.serialize_compressed_video(
            video_data, width, height, num_frames, block_size, levels
        )
        with open(self.file_path, "wb") as f:
            f.write(data_bytes)

    def write_reconstructed_video(self, video_bytes, mode='wb'):
        """
        Escribe los bytes del video reconstruido (YUV plano) en el archivo.
        Por defecto sobrescribe el archivo (modo 'wb'). Si se desea agregar, usar mode='ab'.
        """
        with open(self.file_path, mode) as f:
            f.write(video_bytes)

    def read_header(self, header_size:int = 8):
        """
        Lee el header del archivo comprimido.
        Devuelve: (width, height, num_frames, block_size, levels)
        """
        with open(self.file_path, 'rb') as f:
            header = f.read(header_size)
            width, height, num_frames, block_size, levels = struct.unpack('<HHHBB', header)
        return width, height, num_frames, block_size, levels

    def read_compressed_video(self):
        """
        Lee el archivo comprimido y devuelve la estructura deserializada.
        """
        with open(self.file_path, "rb") as f:
            data = f.read()
        return self.deserialize_compressed_video(data)
    
    def read_original_video(self):
        """
        Lee el resto del archivo (después del header).
        Devuelve los bytes de video original.
        """
        return np.fromfile(self.file_path, dtype=np.uint8)
    
    def serialize_compressed_video(self,compressed_frames, width, height, num_frames, block_size, levels):
        """
        Serializa la estructura de salida de process_video_compression a un objeto bytes,
        usando el mismo formato que BitWriter.write_compressed_video.
        """
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
        Deserializa un archivo binario comprimido y reconstruye la estructura:
        [ [ (processed_blocks, mode_flags, min_vals, steps), ... ], ... ]
        Devuelve: (frames, width, height, num_frames, block_size, levels)
        """
        offset = 0
        width, height, num_frames, block_size, levels = struct.unpack_from('<HHHBB', data, offset)
        offset += struct.calcsize('<HHHBB')
        frames = []
        for _ in range(num_frames):
            frame = []
            for _ in range(3):  # Y, U, V
                n_blocks_y, n_blocks_x = struct.unpack_from("ii", data, offset)
                offset += 8
                min_vals = np.frombuffer(data, dtype=np.float32, count=n_blocks_y * n_blocks_x, offset=offset)
                min_vals = min_vals.reshape((n_blocks_y, n_blocks_x))
                offset += 4 * n_blocks_y * n_blocks_x
                steps = np.frombuffer(data, dtype=np.float32, count=n_blocks_y * n_blocks_x, offset=offset)
                steps = steps.reshape((n_blocks_y, n_blocks_x))
                offset += 4 * n_blocks_y * n_blocks_x
                block_size_sq = block_size * block_size
                num_blocks = n_blocks_y * n_blocks_x * block_size_sq
                processed_blocks = np.frombuffer(data, dtype=np.uint8, count=num_blocks, offset=offset)
                processed_blocks = processed_blocks.reshape((n_blocks_y, n_blocks_x, block_size, block_size))
                offset += num_blocks
                mode_flags = np.frombuffer(data, dtype=np.uint8, count=n_blocks_y * n_blocks_x, offset=offset)
                mode_flags = mode_flags.reshape((n_blocks_y, n_blocks_x))
                offset += n_blocks_y * n_blocks_x
                frame.append((processed_blocks, mode_flags, min_vals, steps))
            frames.append(frame)
        return frames, width, height, num_frames, block_size, levels

