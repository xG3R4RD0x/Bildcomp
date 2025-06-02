import math
import struct
import sys
from typing import Dict, Self, Tuple
from numba import njit


##### utils #####

class BitWriter:
    current_value: int = 0
    significant_bits: int = 0
    output_bytes: bytearray = bytearray()

    def write(self, code: int, num_bits: int):
        assert code < (1 << num_bits)
        free_bits = 8 - self.significant_bits
        while num_bits >= free_bits:
            # code would overflow this.current_value, thus let's write to output_bytes instead
            new_bits = (code >> (num_bits - free_bits)) & ((1 << free_bits) - 1)
            old_bits = self.current_value & ((1 << self.significant_bits) - 1)
            byte = (old_bits << free_bits) | new_bits
            self.output_bytes += bytes([byte])

            self.current_value = 0
            self.significant_bits = 0

            # code >>= free_bits
            num_bits -= free_bits
            free_bits = 8

        # store the left over signifcant_bits of code in self.current_value
        self.current_value = (self.current_value << num_bits) | (code & ((1 << num_bits) - 1))
        self.significant_bits += num_bits

    def flush(self):
        assert self.significant_bits < 8
        if self.significant_bits > 0:
            byte = self.current_value << (8 - self.significant_bits)
            self.output_bytes.append(byte)

class BitReader:
    current_value: int = 0
    significant_bits: int = 0
    input_bytes: bytes
    byte_offset: int = 0

    def __init__(self, data: bytes):
        self.input_bytes = data

    def read(self, num_bits: int) -> int | None:
        bits = self.peek(num_bits)
        self.current_value &= ((1 << (self.significant_bits - num_bits)) - 1)
        self.significant_bits -= num_bits
        return bits

    def peek(self, num_bits: int) -> int | None:
        bits_missing = num_bits - self.significant_bits
        while bits_missing > 0:
            byte = self.input_bytes[self.byte_offset]
            self.byte_offset += 1
            self.current_value = (self.current_value << 8) | byte
            self.significant_bits += 8
            bits_missing -= 8

        bits = (self.current_value >> (self.significant_bits - num_bits)) & ((1 << num_bits) - 1)
        return bits

    def read_huffman(self, sorted_bitlength_set: list[int], bitlengths: list[int], codes: list[int]) -> int:
        for l in sorted_bitlength_set:
            if l == 0:
                continue
            for i, length in enumerate(bitlengths):
                if length == l and self.peek(length) == codes[i]:
                    self.read(length)
                    return i
                
        assert False

    def read_float(self) -> float | None:
        additional_bytes_needed = ((32 - self.significant_bits) + 7) // 8 
        if additional_bytes_needed > len(self.input_bytes[self.byte_offset:]):
            return None

        float_bits = self.peek(32)
        if float_bits is None:
            return None
        float_bytes = int.to_bytes(float_bits, 4, 'big')
        float_value = struct.unpack('<f', float_bytes)
        float_bits = self.read(32)
        return float_value[0]

class RawYUVMetadata:
    width: int
    height: int
    fps: int

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps

def parse_raw_metadata_from_filename(filename: str) -> RawYUVMetadata | None:
    dimensions_str = None
    fps_str = None

    filename_no_ext = filename.split('.', 1)[0]
    for part in filename_no_ext.split('_'):
        if len(part) == 0 or not part[0].isdigit():
            continue

        # we have a digit, so we assume part to be either
        #   1. the dimensions in the form "<width>x<height>"
        #   2. the number of frames per second
        if dimensions_str is None:
            dimensions_str = part
        elif fps_str is None:
            fps_str = part
            
    if dimensions_str is None:
        return None

    width, height = [int(x) for x in dimensions_str.split('x')]
    
    fps = 25
    if fps_str != None:
        fps = int(fps_str)

    return RawYUVMetadata(width, height, fps)

class YUVImage:
    width: int
    height: int

    y: bytes
    u: bytes
    v: bytes

    def __init__(self, width: int, height: int, y: bytes, u: bytes, v: bytes):
        self.width = width
        self.height = height

        self.y = y
        self.u = u
        self.v = v

    def to_rgb_bytes(self) -> bytes:
        data = bytes(self.width * self.height * 3)
        for i in range(self.width * self.height):
            pos_x = i % self.width
            pos_y = i // self.width
            pos_y = self.height - pos_y - 1

            # print(len(self.u), pos_x // 2 + pos_y // 2 * self.width // 2)

            i_y = self.y[pos_y * self.width + pos_x]
            i_u = self.u[pos_x // 2 + pos_y // 2 * self.width // 2]
            i_v = self.v[pos_x // 2 + pos_y // 2 * self.width // 2]

            # i_y = i_y / 8 * 8;
            # i_u = (i_u - 128) / 8 * 8 + 128;
            # i_v = (i_v - 128) / 8 * 8 + 128;

            y = i_y
            u = i_u
            v = i_v

            R = y + 1.402 * (v - 128)
            G = y - 0.344 * (u - 128) - 0.714 * (v - 128)
            B = y + 1.772 * (u - 128)

            R = max(min(R, 255), 0)
            G = max(min(G, 255), 0)
            B = max(min(B, 255), 0)

            data[i * 3 + 0] = R
            data[i * 3 + 1] = G
            data[i * 3 + 2] = B
        return data

    def tga_data(self) -> list[int]:
        tga_header = [0] * 18
        tga_header[2] = 2 # uncompressed RGB
        tga_header[12] = self.width & 0xff
        tga_header[13] = (self.width >> 8) & 0xff
        tga_header[14] = self.height & 0xff
        tga_header[15] = (self.height >> 8) & 0xff
        tga_header[16] = 24  # 24 bits per pixel

        tga_data = []
        for i in range(self.width * self.height):
            pos_x = i % self.width
            pos_y = i // self.width
            pos_y = self.height - pos_y - 1

            # print(len(self.u), pos_x // 2 + pos_y // 2 * self.width // 2)

            i_y = self.y[pos_y * self.width + pos_x]
            i_u = self.u[pos_x // 2 + pos_y // 2 * self.width // 2]
            i_v = self.v[pos_x // 2 + pos_y // 2 * self.width // 2]

            # i_y = i_y / 8 * 8;
            # i_u = (i_u - 128) / 8 * 8 + 128;
            # i_v = (i_v - 128) / 8 * 8 + 128;

            y = i_y
            u = i_u
            v = i_v

            R = y + 1.402 * (v - 128)
            G = y - 0.344 * (u - 128) - 0.714 * (v - 128)
            B = y + 1.772 * (u - 128)

            R = max(min(R, 255), 0)
            G = max(min(G, 255), 0)
            B = max(min(B, 255), 0)

            tga_data.extend(int(x) for x in [B, G, R])
        return tga_header + tga_data

    def save_as_tga(self, output_filename: str):
        data = self.tga_data()
        with open(output_filename, 'wb') as f:
                f.write(bytes(data))


class Vid:
    fps: int
    frame_width: int
    frame_height: int
    
    quantization_y_levels: int
    quantization_u_levels: int
    quantization_v_levels: int

    frames: list[YUVImage]

    def __init__(self, fps: int, frame_width: int, frame_height: int, frames: list[YUVImage]) -> Self:
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.quantization_u_levels = 128
        self.quantization_v_levels = 128
        self.quantization_y_levels = 128

        self.frames = frames

    def read_from_file(filename: str) -> Self:
        content = None
        with open(filename, 'rb') as f:
            content = f.read()
        
        if content[:4] != b'.VID' or content[4] != 0:
            raise ValueError("Invalid file format")
        

        fps = content[5]
        frame_width = int.from_bytes(content[6:8], 'little')
        frame_height = int.from_bytes(content[8:10], 'little')

        self = Vid(fps, frame_width, frame_height, [])

        self.quantization_y_levels = content[10]
        self.quantization_u_levels = content[11]
        self.quantization_v_levels = content[12]

        bitreader = BitReader(content[13:])

        def read_frame_data(bitreader: BitReader, width: int, height: int, quantization_levels: int) -> list[int]:
            float_step = bitreader.read_float()
            # print(f"{float_step = }")
            if float_step is None:
                return None

            float_min = bitreader.read_float()

            # step = (2 * float_range) / quantization_levels

            bitlengths = []
            for i in range(quantization_levels):
                bitlen = bitreader.read(4)
                bitlengths.append(bitlen)
            huffman_codes = huffman_codes_from_bit_lengths(bitlengths)
            sorted_bitlength_set = sorted(set(bitlengths))
            # print(f"{bitlengths = }")
            # print(f"{huffman_codes = }")
            # y_length_code_pairs = list(zip(bitlengths, huffman_codes))
            # print(f"{y_length_code_pairs = }")

            dequantized_data = []
            for i in range(width * height):
                # print(f"reading y value: {i} of {width * height}")
                quantized_value = bitreader.read_huffman(sorted_bitlength_set, bitlengths, huffman_codes)

                # print(f"{quantized_value = }")
                dequantized_value = (quantized_value * float_step) + float_min
                dequantized_data.append(dequantized_value)

            data = inverse_discrete_cosine_transform_2d(width, height, width, dequantized_data)
            return data
        
        # print(self.frame_width, self.frame_height, self.frame_width * self.frame_height)
        while y_data := read_frame_data(bitreader, self.frame_width, self.frame_height, self.quantization_y_levels):
            u_data = read_frame_data(bitreader, self.frame_width // 2, self.frame_height // 2, self.quantization_u_levels)
            assert u_data is not None
            v_data = read_frame_data(bitreader, self.frame_width // 2, self.frame_height // 2, self.quantization_v_levels)
            assert v_data is not None

            frame = YUVImage(self.frame_width, self.frame_height, y_data, u_data, v_data)
            self.frames.append(frame)
            print(f"frame: {len(self.frames)}\r", end='')

        print(f"frame: {len(self.frames)}")
        return self

    def save_to_file(self, filename: str):
        # TODO:
        #  - determine appropriate values for X_range for every frame from self.frames
        #  - serialize all frames into a bitstream

        frames_bitwriter = BitWriter()
        def write_frame_data(fstep: float, fmin: float, length_code_pairs: list[Tuple[int, int]], data: list[int]):
            for byte in struct.pack('<f', fstep):
                frames_bitwriter.write(byte, 8)

            for byte in struct.pack('<f', fmin):
                frames_bitwriter.write(byte, 8)

            for bitlen, _ in length_code_pairs:
                assert bitlen < (1 << 4)
                frames_bitwriter.write(bitlen, 4)

            for i, x in enumerate(data):
                length, code = length_code_pairs[x]
                frames_bitwriter.write(code, length)

        for i, frame in enumerate(self.frames):
            print(f"Writing frame: {i}\r", end="")
            # decorrelation
            transformed_y = discrete_cosine_transform_2d(self.frame_width, self.frame_height, self.frame_width, frame.y)
            transformed_u = discrete_cosine_transform_2d(self.frame_width // 2, self.frame_height // 2, self.frame_width // 2, frame.u)
            transformed_v = discrete_cosine_transform_2d(self.frame_width // 2, self.frame_height // 2, self.frame_width // 2, frame.v)

            # quantization
            y_range = max(abs(x) for x in transformed_y)
            u_range = max(abs(x) for x in transformed_u)
            v_range = max(abs(x) for x in transformed_v)

            y_max = max(transformed_y)
            u_max = max(transformed_u)
            v_max = max(transformed_v)

            y_min = min(transformed_y)
            u_min = min(transformed_u)
            v_min = min(transformed_v)

            y_range = y_max - y_min
            u_range = u_max - u_min
            v_range = v_max - v_min

            y_step = y_range / self.quantization_y_levels
            u_step = u_range / self.quantization_u_levels
            v_step = v_range / self.quantization_v_levels
            
            y_quantized = [min(int((x - y_min) // y_step), self.quantization_y_levels - 1) for x in transformed_y]
            u_quantized = [min(int((x - u_min) // u_step), self.quantization_u_levels - 1) for x in transformed_u]
            v_quantized = [min(int((x - v_min) // v_step), self.quantization_v_levels - 1) for x in transformed_v]

            # probability modelling
            y_occurences = occurence_distribution(y_quantized)
            u_occurences = occurence_distribution(u_quantized)
            v_occurences = occurence_distribution(v_quantized)

            y_symbols = []
            y_symbol_counts = []
            for k, v in y_occurences.items():
                y_symbols.append(k)
                y_symbol_counts.append(v)

            u_symbols = []
            u_symbol_counts = []
            for k, v in u_occurences.items():
                u_symbols.append(k)
                u_symbol_counts.append(v)

            v_symbols = []
            v_symbol_counts = []
            for k, v in v_occurences.items():
                v_symbols.append(k)
                v_symbol_counts.append(v)

            # entropy coding
            y_bitlengths = huffman_coding_length_limited(y_symbol_counts, 15)
            u_bitlengths = huffman_coding_length_limited(u_symbol_counts, 15)
            v_bitlengths = huffman_coding_length_limited(v_symbol_counts, 15)

            y_bitlengths_all = []
            for i in range(self.quantization_y_levels):
                bitlen = 0
                if i in y_symbols:
                    index = y_symbols.index(i)
                    bitlen = y_bitlengths[index]
                y_bitlengths_all.append(bitlen)

            u_bitlengths_all = []
            for i in range(self.quantization_u_levels):
                bitlen = 0
                if i in u_symbols:
                    index = u_symbols.index(i)
                    bitlen = u_bitlengths[index]
                u_bitlengths_all.append(bitlen)

            v_bitlengths_all = []
            for i in range(self.quantization_v_levels):
                bitlen = 0
                if i in v_symbols:
                    index = v_symbols.index(i)
                    bitlen = v_bitlengths[index]
                v_bitlengths_all.append(bitlen)

            y_codes = huffman_codes_from_bit_lengths(y_bitlengths_all)
            u_codes = huffman_codes_from_bit_lengths(u_bitlengths_all)
            v_codes = huffman_codes_from_bit_lengths(v_bitlengths_all)

            y_length_code_pairs = list(zip(y_bitlengths_all, y_codes))
            u_length_code_pairs = list(zip(u_bitlengths_all, u_codes))
            v_length_code_pairs = list(zip(v_bitlengths_all, v_codes))
            
            write_frame_data(y_step, y_min, y_length_code_pairs, y_quantized)

                # print(f"{y_bitlengths = }")
                # print(f"{y_occurences = }")
                # print(f"{y_symbols = }")
                # print(f"{y_length_code_pairs = }")


            write_frame_data(u_step, u_min, u_length_code_pairs, u_quantized)
            write_frame_data(v_step, v_min, v_length_code_pairs, v_quantized)
            
        
        print(f"Written all {len(self.frames)} frames")
        frames_bitwriter.flush()

        with open(filename, "wb") as f:
            f.write(b".VID")
            f.write(int.to_bytes(0, 1, 'little'))

            f.write(int.to_bytes(self.fps, 1, 'little'))
            f.write(int.to_bytes(self.frame_width, 2, 'little'))
            f.write(int.to_bytes(self.frame_height, 2, 'little'))

            f.write(int.to_bytes(self.quantization_y_levels, 1, 'little'))
            f.write(int.to_bytes(self.quantization_u_levels, 1, 'little'))
            f.write(int.to_bytes(self.quantization_v_levels, 1, 'little'))

            # f.write(bytes(struct.pack('<f', self._quantization_y_range)))
            # f.write(bytes(struct.pack('<f', self._quantization_u_range)))
            # f.write(bytes(struct.pack('<f', self._quantization_v_range)))

            f.write(frames_bitwriter.output_bytes)
        

##### decorrelation #####

def discrete_cosine_transform_row(x: list[int]) -> list[float]:
    N = len(x)
    Xs = []
    for k in range(N):
        C0 = 1/math.sqrt(2) if k == 0 else 1
        
        summed_xs = 0
        for n in range(N):
            summed_xs += x[n] * math.cos((2 * n + 1) * k * math.pi / (2 * N))
        
        X = C0 * math.sqrt(2/N) * summed_xs
        Xs.append(X)
    return Xs

def inverse_discrete_cosine_transform_row(X: list[float]) -> list[int]:
    N = len(X)
    xs = []
    for n in range(N):
        summed_Xs = 0
        for k in range(N):
            C0 = 1/math.sqrt(2) if k == 0 else 1
            summed_Xs += C0 * X[k] * math.cos((2 * n + 1) * k * math.pi / (2 * N))
        
        x = math.sqrt(2/N) * summed_Xs
        xs.append(round(x))
    return xs

@njit
def precompute_dct_cosine_values(N: int) -> list[float]:
    values: list[float] = []
    for k in range(N):
        for n in range(N):
            v = math.cos((2 * n + 1) * k * math.pi / (2 * N))
            values.append(v)
    return values


@njit
def discrete_cosine_transform_2d(width: int, height: int, stride: int, data: list[int]) -> list[float]:
    # transform horizontally
    transformed_horizontally = []
    N = width

    precomputed = precompute_dct_cosine_values(N)
    sqrt2_repro = 1/math.sqrt(2)
    sqrt2_over_N = math.sqrt(2/N)

    for y in range(height):
        for k in range(N):
            C0 = sqrt2_repro if k == 0 else 1

            summed_xs = 0
            for x in range(width):
                offset = (y * stride) + x
                n = x
                summed_xs += data[offset] * precomputed[k * N + n]
            
            X = C0 * sqrt2_over_N * summed_xs
            transformed_horizontally.append(X)

    # transform vertically
    # transformed = [0] * width * height
    # N = height
    # for x in range(width):
    #     for k in range(N):
    #         C0 = 1/math.sqrt(2) if k == 0 else 1
            
    #         summed_xs = 0
    #         for y in range(height):
    #             offset = (y * stride) + x
    #             n = y
    #             summed_xs += transformed_horizontally[offset] * math.cos((2 * n + 1) * k * math.pi / (2 * N))
            
    #         X = C0 * math.sqrt(2/N) * summed_xs
    #         offset = (k * width) + x # NOTE: we are intentionally using k and width instead of stride, because this is in terms of the output coefficients
    #         transformed[offset] = X

    # return transformed
    return transformed_horizontally

@njit
def inverse_discrete_cosine_transform_2d(width: int, height: int, stride: int, data: list[float]) -> list[int]:
    # inverse transform vertically
    # reconstructed_vertically = [0] * width * height
    # N = height
    # for x in range(width):
    #     for n in range(N):
            
    #         summed_xs = 0
    #         for y in range(height):
    #             offset = (y * stride) + x
    #             k = y

    #             C0 = 1/math.sqrt(2) if k == 0 else 1
    #             summed_xs += C0 * data[offset] * math.cos((2 * n + 1) * k * math.pi / (2 * N))
            
    #         X = math.sqrt(2/N) * summed_xs
    #         offset = (k * width) + x # NOTE: we are intentionally using k and width instead of stride, because this is in terms of the output coefficients
    #         reconstructed_vertically[offset] = X
    
    # inverse transform horizontally

    reconstructed = []
    N = width

    precomputed = precompute_dct_cosine_values(N)
    sqrt2_repro = 1/math.sqrt(2)
    sqrt2_over_N = math.sqrt(2/N)

    for y in range(height):
        for n in range(N):

            summed_xs = 0
            for x in range(width):
                offset = (y * stride) + x
                k = x

                C0 = sqrt2_repro if k == 0 else 1
                summed_xs += C0 * data[offset] * precomputed[k * N + n]
                # summed_xs += C0 * data[offset] * math.cos((2 * n + 1) * k * math.pi / (2 * N))
            
            X = sqrt2_over_N * summed_xs
            reconstructed.append(X)
    return reconstructed


##### probability modelling #####

def occurence_distribution(data: list[int]) -> Dict[int, int]:
    occurences = dict()
    for x in data:
        if x not in occurences:
            occurences[x] = 1
        else:
            occurences[x] += 1

    return occurences

def entropy(occurences: list[int]) -> float:
    s = sum(occurences)
    return -sum([x/s * math.log(x/s, 2) for x in occurences])

##### entropy coding #####

def huffman_coding_length_limited(sorted_occurenes: list[int], max_bits: int) -> list[int]:
    # package-merge algorithm
    # TODO: add reference to paper + useful explanations

    # assert not any(x == 0 for x in sorted_occurenes)
    # assert max_bits > 0
    # assert len(sorted_occurenes) > 1
    # assert (1 << max_bits) >= len(sorted_occurenes)

    initial_packages: list[Tuple[int, list[int]]] = []
    for i, occ in enumerate(sorted_occurenes):
        initial_packages.append((occ, [i]))
    initial_packages = sorted(initial_packages, key=lambda p: (p[0], -len(p[1])))

    packages = initial_packages.copy()
    for i in range(max_bits - 1):
        new_packages = []

        for i in range(len(packages) // 2):
            p0: Tuple[list[int], int] = packages[i * 2 + 0] # every even indexed item
            p1: Tuple[list[int], int] = packages[i * 2 + 1] # every  odd indexed item

            items = []
            items.extend(p0[1])
            items.extend(p1[1])
            p0p1 = (p0[0] + p1[0], items)
            new_packages.append(p0p1)

        packages = new_packages + initial_packages
        packages = sorted(packages, key=lambda p: (p[0], -len(p[1])))

    num_packages = 2 * len(sorted_occurenes) - 2
    symbols = [p[1] for p in packages[:num_packages]]
    symbols_flattened = [s for syms in symbols for s in syms]

    bit_lengths = []
    for i in range(len(sorted_occurenes)):
        bit_lengths.append(symbols_flattened.count(i))

    return bit_lengths

# @njit
def huffman_codes_from_bit_lengths(bit_lengths: list[int]) -> list[int]:
    # https://www.rfc-editor.org/rfc/rfc1951.html#section-3.2.2
    MAX_BITS = max(bit_lengths)

    bl_count = []
    for i in range(0, max(bit_lengths)):
        bl_count.append(bit_lengths.count(i))

    next_code = [0]
    bl_count[0] = 0
    code = 0
    for bits in range(1, MAX_BITS + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code.append(code)

    codes = []
    for length in bit_lengths:
        if length != 0:
            codes.append(next_code[length])
            next_code[length] += 1
        else:
            codes.append(0)

    return codes

##### main logic #####
import time

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        exit(f"Usage: ./{args[0]} <filename>")

    filename = args[1]
    content = None
    with open(filename, 'rb') as f:
        content = f.read()

    metadata = parse_raw_metadata_from_filename(filename)

    ys_size = metadata.width * metadata.height * 1
    us_size = metadata.width * metadata.height // 4
    vs_size = metadata.width * metadata.height // 4

    frame_size = ys_size + us_size + vs_size

    # tga_frames = []
    frames = []
    offset = 0
    while len(content) - offset >= frame_size:
        print(f"\rProcessing frame: {len(frames)}", end="")
        ys = content[offset:offset+ys_size]
        us = content[offset+ys_size:offset+ys_size+us_size]
        vs = content[offset+ys_size+us_size:offset+ys_size+us_size+vs_size]
        offset += frame_size

        frame = YUVImage(metadata.width, metadata.height, ys, us, vs)
        frames.append(frame)
        # tga_frames.append(frame.tga_data())

        # if len(frames) == 1:
        #     occ1 = occurence_distribution(frames[0].y)
        #     print(f"{entropy(occ1) = }")
        #     plt.hist(frames[0].y, bins=256)
        #     plt.show()

    print("")

    # for frame in frames:
    #     frame.save_as_tga("output.tga")
    #     time.sleep(1 / metadata.fps)

    # tga_frames = [frame.tga_data() for frame in frames]
    # for frame in tga_frames:
    #     with open("output.tga", 'wb') as f:
    #         f.write(bytes(frame))
    #     time.sleep(1 / metadata.fps)

    print(f"{metadata.width = }")
    print(f"{metadata.height = }")
    print(f"{metadata.fps = }")

    vid = Vid(metadata.fps, metadata.width, metadata.height, frames)
    vid.save_to_file("output.vid")

    vid = Vid.read_from_file("output.vid")

    with open('output/output.yuv', 'wb') as f:
        for frame in vid.frames:
            f.write(bytes([max(0, min(int(x), 255)) for x in frame.y]))
            f.write(bytes([max(0, min(int(x), 255)) for x in frame.u]))
            f.write(bytes([max(0, min(int(x), 255)) for x in frame.v]))

    # frames = vid.frames
    # assert len(frames) > 0
    # tga_frames = [frame.tga_data() for frame in frames]
    # for frame in tga_frames:
    #     with open("output.tga", 'wb') as f:
    #         f.write(bytes(frame))
    #     time.sleep(1 / vid.fps)