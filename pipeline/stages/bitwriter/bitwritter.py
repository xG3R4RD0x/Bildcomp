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
    
    def serialize_compressed_video(self, compressed_frames, width, height, num_frames, block_size, levels):
        """
        Serializes the output structure of process_video_compression to a bytes object,
        now supporting Huffman: encoded_blocks, huff_table, pads, shape, mode_flags, min_vals, steps
        Huffman table is stored as a compact list of (symbol, code as bytes, code bitlength)
        """
        def serialize_huff_table(huff_table):
            # Returns: bytes: [num_entries][symbol][code_len][code_bytes]...
            entries = list(huff_table.items())
            out = [struct.pack('H', len(entries))]  # num_entries (uint16)
            for symbol, code in entries:
                code_len = len(code)
                code_int = int(code, 2) if code else 0
                n_bytes = (code_len + 7) // 8
                out.append(struct.pack('B', symbol))  # symbol as uint8
                out.append(struct.pack('B', code_len))  # code length in bits
                out.append(code_int.to_bytes(n_bytes, 'big'))  # code bits
            return b''.join(out)

        byte_chunks = []
        # Header: width, height, num_frames, block_size, levels (all <HHHBB>)
        byte_chunks.append(struct.pack('<HHHBB', width, height, num_frames, block_size, levels))
        for frame in compressed_frames:
            for channel_tuple in frame:  # [y, u, v] tuples
                encoded_blocks, huff_table, pads, shape, mode_flags, min_vals, steps = channel_tuple
                n_blocks_y, n_blocks_x, block_size = shape
                # Save shape (3 int32)
                byte_chunks.append(struct.pack("iii", n_blocks_y, n_blocks_x, block_size))
                # Save Huffman table (compact)
                huff_table_bytes = serialize_huff_table(huff_table)
                byte_chunks.append(struct.pack("I", len(huff_table_bytes)))
                byte_chunks.append(huff_table_bytes)
                # Save pads (as uint8 array)
                pads_arr = np.array(pads, dtype=np.uint8)
                byte_chunks.append(struct.pack("I", len(pads_arr)))
                byte_chunks.append(pads_arr.tobytes())
                # Save encoded_blocks (each as bytes, with length)
                byte_chunks.append(struct.pack("I", len(encoded_blocks)))
                for b in encoded_blocks:
                    byte_chunks.append(struct.pack("I", len(b)))
                    byte_chunks.append(b)
                # Save min_vals and steps (float32 per block)
                byte_chunks.append(min_vals.astype(np.float32).tobytes())
                byte_chunks.append(steps.astype(np.float32).tobytes())
                # Save all mode_flags (uint8)
                byte_chunks.append(mode_flags.astype(np.uint8).tobytes())
        return b"".join(byte_chunks)

    def deserialize_compressed_video(self, data: bytes):
        """
        Deserializes a compressed binary file and reconstructs the structure:
        [ [ (encoded_blocks, huff_table, pads, shape, mode_flags, min_vals, steps), ... ], ... ]
        Returns: (frames, width, height, num_frames, block_size, levels)
        """
        def deserialize_huff_table(b):
            # Returns: dict {symbol: bitstring}
            offset = 0
            num_entries = struct.unpack_from('H', b, offset)[0]
            offset += 2
            huff_table = {}
            for _ in range(num_entries):
                symbol = struct.unpack_from('B', b, offset)[0]
                offset += 1
                code_len = struct.unpack_from('B', b, offset)[0]
                offset += 1
                n_bytes = (code_len + 7) // 8
                code_bytes = b[offset:offset+n_bytes]
                offset += n_bytes
                code_int = int.from_bytes(code_bytes, 'big') if n_bytes > 0 else 0
                code = bin(code_int)[2:].zfill(code_len) if code_len > 0 else ''
                huff_table[symbol] = code
            return huff_table

        offset = 0
        width, height, num_frames, block_size, levels = struct.unpack_from('<HHHBB', data, offset)
        offset += struct.calcsize('<HHHBB')
        frames = []
        for _ in range(num_frames):
            frame = []
            for _ in range(3):  # Y, U, V
                n_blocks_y, n_blocks_x, block_size_r = struct.unpack_from("iii", data, offset)
                offset += 12
                # Huffman table
                huff_table_len = struct.unpack_from("I", data, offset)[0]
                offset += 4
                huff_table = deserialize_huff_table(data[offset:offset+huff_table_len])
                offset += huff_table_len
                # Pads
                pads_len = struct.unpack_from("I", data, offset)[0]
                offset += 4
                pads = np.frombuffer(data, dtype=np.uint8, count=pads_len, offset=offset).tolist()
                offset += pads_len
                # Encoded blocks
                num_blocks = struct.unpack_from("I", data, offset)[0]
                offset += 4
                encoded_blocks = []
                for _ in range(num_blocks):
                    blen = struct.unpack_from("I", data, offset)[0]
                    offset += 4
                    encoded_blocks.append(data[offset:offset+blen])
                    offset += blen
                # min_vals, steps
                min_vals = np.frombuffer(data, dtype=np.float32, count=n_blocks_y * n_blocks_x, offset=offset)
                min_vals = min_vals.reshape((n_blocks_y, n_blocks_x))
                offset += 4 * n_blocks_y * n_blocks_x
                steps = np.frombuffer(data, dtype=np.float32, count=n_blocks_y * n_blocks_x, offset=offset)
                steps = steps.reshape((n_blocks_y, n_blocks_x))
                offset += 4 * n_blocks_y * n_blocks_x
                # mode_flags
                mode_flags = np.frombuffer(data, dtype=np.uint8, count=n_blocks_y * n_blocks_x, offset=offset)
                mode_flags = mode_flags.reshape((n_blocks_y, n_blocks_x))
                offset += n_blocks_y * n_blocks_x
                shape = (n_blocks_y, n_blocks_x, block_size_r)
                frame.append((encoded_blocks, huff_table, pads, shape, mode_flags, min_vals, steps))
            frames.append(frame)
        return frames, width, height, num_frames, block_size, levels

