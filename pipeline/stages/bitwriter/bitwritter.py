import struct
import numpy as np

class BitWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def write_header(self, width, height, num_frames, block_size, levels):
        """
        Writes the header of the compressed file.
        Header: [width (H)][height (H)][num_frames (H)][block_size (B)][levels (B)]
        """
        with open(self.file_path, 'wb') as f:
            f.write(struct.pack('<HHHBB', width, height, num_frames, block_size, levels))
                    
    def write_compressed_video(self, video_data, width, height, num_frames, block_size, levels):
        """
        Writes the processed frames (arrays and metadata) to the file.
        """
        data_bytes = self.serialize_compressed_video(
            video_data, width, height, num_frames, block_size, levels
        )
        with open(self.file_path, "wb") as f:
            f.write(data_bytes)

    def write_reconstructed_video(self, video_bytes, mode='wb'):
        """
        Writes the bytes of the reconstructed video (raw YUV) to the file.
        By default, it overwrites the file (mode 'wb'). If you want to append, use mode='ab'.
        """
        with open(self.file_path, mode) as f:
            f.write(video_bytes)

    def read_header(self, header_size:int = 8):
        """
        Reads the header of the compressed file.
        Returns: (width, height, num_frames, block_size, levels)
        """
        with open(self.file_path, 'rb') as f:
            header = f.read(header_size)
            width, height, num_frames, block_size, levels = struct.unpack('<HHHBB', header)
        return width, height, num_frames, block_size, levels

    def read_compressed_video(self):
        """
        Reads the compressed file and returns the deserialized structure.
        """
        with open(self.file_path, "rb") as f:
            data = f.read()
        return self.deserialize_compressed_video(data)
    
    def read_original_video(self):
        """
        Reads the rest of the file (after the header).
        Returns the bytes of the original video.
        """
        return np.fromfile(self.file_path, dtype=np.uint8)
    
    def serialize_compressed_video(self,compressed_frames, width, height, num_frames, block_size, levels):
        """
        Serializes the output structure of process_video_compression to a bytes object,
        using the same format as BitWriter.write_compressed_video.
        """
        byte_chunks = []
        # Header: width, height, num_frames, block_size, levels (all <HHHBB>)
        byte_chunks.append(struct.pack('<HHHBB', width, height, num_frames, block_size, levels))
        for frame in compressed_frames:
            for channel_tuple in frame:  # [y_out, u_out, v_out]
                processed_blocks, mode_flags, min_vals, steps = channel_tuple
                n_blocks_y, n_blocks_x = mode_flags.shape
                # Save shape (2 int32)
                byte_chunks.append(struct.pack("ii", n_blocks_y, n_blocks_x))
                # Save min_vals and steps (float32 per block)
                byte_chunks.append(min_vals.astype(np.float32).tobytes())
                byte_chunks.append(steps.astype(np.float32).tobytes())
                # Save all blocks (uint8)
                byte_chunks.append(processed_blocks.astype(np.uint8).tobytes())
                # Save all mode_flags (uint8)
                byte_chunks.append(mode_flags.astype(np.uint8).tobytes())
        return b"".join(byte_chunks)

    def deserialize_compressed_video(self, data: bytes):
        """
        Deserializes a compressed binary file and reconstructs the structure:
        [ [ (processed_blocks, mode_flags, min_vals, steps), ... ], ... ]
        Returns: (frames, width, height, num_frames, block_size, levels)
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

