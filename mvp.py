import itertools
import math
import struct
import sys
from typing import Dict, Self, Tuple


##### utils #####

class BitWriter:
    current_value: int = 0
    significant_bits: int = 0
    output_bytes: bytes = bytes()

    def write(self, code: int, significant_bits: int):
        free_bits = 8 - self.significant_bits
        while significant_bits >= free_bits:
            # code would overflow this.current_value, thus let's write to output_bytes instead
            new_bits = code & ((1 << free_bits) - 1)
            old_bits = self.current_value & ((1 << self.significant_bits) - 1)
            byte = (new_bits << self.significant_bits) | old_bits
            self.output_bytes += bytes([byte])

            self.current_value = 0
            self.significant_bits = 0

            code >>= free_bits
            significant_bits -= free_bits
            free_bits = 8

        # store the left over signifcant_bits of code in self.current_value
        self.current_value = (code & ((1 << significant_bits) - 1))

    def flush(self):
        if self.significant_bits > 0:
            self.output_bytes += bytes([self.current_value])

class BitReader:
    current_value: int = 0
    significant_bits: int = 0
    input_bytes: bytes
    byte_offset: int = 0

    def __init__(self, data: bytes):
        self.input_bytes = data

    def read(self, num_bits: int) -> int:
        pass

    def peek(self, num_bits: int) -> int:
        pass

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

    y: list[int]
    u: list[int]
    v: list[int]

    def __init__(self, width: int, height: int, y: list[int], u: list[int], v: list[int]):
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
        data = self.tga_data
        with open(output_filename, 'wb') as f:
                f.write(bytes(data))

class Vid:
    fps: int
    frame_width: int
    frame_height: int
    
    quantization_y_levels: int
    quantization_u_levels: int
    quantization_v_levels: int

    # _quantization_y_range: int | None
    # _quantization_u_range: int | None
    # _quantization_v_range: int | None

    frames: list[YUVImage]

    def __init__(self, fps: int, frame_width: int, frame_height: int, frames: list[YUVImage]) -> Self:
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.quantization_u_levels = 64
        self.quantization_v_levels = 64
        self.quantization_y_levels = 64

        self.frames = frames

    def read_from_file(filename: str) -> Self:
        content = None
        with open(filename, 'rb') as f:
            content = f.read()
        
        if content[:4] != b'.VID' or content[4] != 0:
            raise ValueError("Invalid file format")
        
        self = Vid()

        self.fps = content[5]
        self.frame_width = int.from_bytes(content[6:9], 2, 'little')
        self.frame_height = int.from_bytes(content[8:10], 2, 'little')

        self.quantization_y_levels = content[10]
        self.quantization_u_levels = content[11]
        self.quantization_v_levels = content[12]

        self.frames = []
        bitreader = BitReader(content[13:])

        # self._quantization_y_range(struct.unpack('<f', content[13:17]))
        # self._quantization_u_range(struct.unpack('<f', content[17:21]))
        # self._quantization_v_range(struct.unpack('<f', content[21:25]))

        # TODO: parse data
        # For every frame:
        #  - read and parse alphabet lengths for huffman coding
        #  - read and interpret the data for the frame
        #  - append the frame to self.frames

        return self
    
    def save_to_file(self, filename: str):
        # TODO:
        #  - determine appropriate values for X_range for every frame from self.frames
        #  - serialize all frames into a bitstream
        frames_bitwriter = BitWriter()
        for frame in frames:
            # decorrelation
            transformed_y = discrete_cosine_transform_2d(self.frame_width, self.frame_height, self.frame_width, self.frames[0].y)
            transformed_u = discrete_cosine_transform_2d(self.frame_width // 2, self.frame_height // 2, self.frame_width // 2, self.frames[0].u)
            transformed_v = discrete_cosine_transform_2d(self.frame_width // 2, self.frame_height // 2, self.frame_width // 2, self.frames[0].v)

            # quantization
            y_range = max(abs(x) for x in transformed_y)
            u_range = max(abs(x) for x in transformed_u)
            v_range = max(abs(x) for x in transformed_v)

            y_step = (2 * y_range) / self.quantization_y_levels
            u_step = (2 * u_range) / self.quantization_u_levels
            v_step = (2 * v_range) / self.quantization_v_levels
            
            y_quantized = [(x + y_range) // y_step for x in transformed_y]
            u_quantized = [(x + y_range) // u_step for x in transformed_u]
            v_quantized = [(x + y_range) // v_step for x in transformed_v]

            # probability modelling
            y_occurences = occurence_distribution(y_quantized)
            u_occurences = occurence_distribution(u_quantized)
            v_occurences = occurence_distribution(v_quantized)

            y_symbols = []
            y_symbol_counts = []
            for k, v in y_occurences.items():
                y_symbols.append(k)
                y_symbol_counts.append(v)

            # entropy coding
            y_bitlengths = huffman_coding_length_limited(y_symbol_counts, 15)
            u_bitlengths = huffman_coding_length_limited([v for k, v in u_occurences.items()], 15)
            v_bitlengths = huffman_coding_length_limited([v for k, v in v_occurences.items()], 15)
            
            y_codes = huffman_codes_from_bit_lengths(y_bitlengths)
            u_codes = huffman_codes_from_bit_lengths(u_bitlengths)
            v_codes = huffman_codes_from_bit_lengths(v_bitlengths)

            for byte in struct.pack('<f', y_range):
                frames_bitwriter.write(byte, 8)

            for x in range(self.quantization_y_levels):
                bitlen = 0
                if x in y_symbols:
                    index = y_symbols.index(x)
                    bitlen = y_bitlengths[index]
                frames_bitwriter.write(bitlen, 4)

            for x in y_quantized:
                index = y_symbols.index(x)
                bitlen = y_bitlengths[index]
                frames_bitwriter.write(y_codes[index], bitlen)
        
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

def discrete_cosine_transform_2d(width: int, height: int, stride: int, data: list[int]) -> list[float]:
    # transform horizontally
    transformed_horizontally = []
    N = width
    for y in range(height):
        for k in range(N):
            C0 = 1/math.sqrt(2) if k == 0 else 1

            summed_xs = 0
            for x in range(width):
                offset = (y * stride) + x
                n = x
                summed_xs += data[offset] * math.cos((2 * n + 1) * k * math.pi / (2 * N))
            
            X = C0 * math.sqrt(2/N) * summed_xs
            transformed_horizontally.append(X)

    # transform vertically
    transformed = [0] * width * height
    N = height
    for x in range(width):
        for k in range(N):
            C0 = 1/math.sqrt(2) if k == 0 else 1
            
            summed_xs = 0
            for y in range(height):
                offset = (y * stride) + x
                n = y
                summed_xs += data[offset] * math.cos((2 * n + 1) * k * math.pi / (2 * N))
            
            X = C0 * math.sqrt(2/N) * summed_xs
            offset = (k * width) + x # NOTE: we are intentionally using k and width instead of stride, because this is in terms of the output coefficients
            transformed[offset] = X

    return transformed

def inverse_discrete_cosine_transform_2d(width: int, height: int, stride: int, data: list[float]) -> list[int]:
    # inverse transform vertically
    
    # inverse transform horizontally
    pass


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

def huffman_coding(sorted_occurences: list[int]) -> list[int]:
    # TODO: do huffman coding
    pass

def huffman_coding_length_limited(sorted_occurenes: list[int], max_bits: int) -> list[int]:
    # package-merge algorithm
    # TODO: add reference to paper + useful explanations

    assert not any(x == 0 for x in sorted_occurenes)
    assert max_bits > 0
    assert len(sorted_occurenes) > 2
    assert (1 << max_bits) >= len(sorted_occurenes)

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

    return codes

##### main logic #####

import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # print(f"{huffman_coding_length_limited([2, 3, 8, 2, 1], 3) = }")
    # print(f"{huffman_coding_length_limited([2, 3, 8, 2, 1], 4) = }")
    # print(f"{huffman_codes_from_bit_lengths([3, 3, 3, 3, 3, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) = }")
    # exit()

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

    tga_frames = []
    frames = []
    while len(content) > frame_size:
        print(f"\rProcessing frame: {len(frames)}", end="")
        ys = list(content[0:ys_size])
        us = list(content[ys_size:][:us_size])
        vs = list(content[ys_size+us_size:][:vs_size])
        content = content[frame_size:]

        frame = YUVImage(metadata.width, metadata.height, ys, us, vs)
        frames.append(frame)
        tga_frames.append(frame.tga_data())

        # if len(frames) == 1:
        #     occ1 = occurence_distribution(frames[0].y)
        #     print(f"{entropy(occ1) = }")
        #     plt.hist(frames[0].y, bins=256)
        #     plt.show()

    print("")

    # print(f"{ = }")
    # for frame in frames:
    #     frame.save_as_tga("output.tga")
    #     time.sleep(1 / metadata.fps)

    tga_frames = [frame.tga_data() for frame in frames]
    for frame in tga_frames:
        with open("output.tga", 'wb') as f:
            f.write(bytes(frame))
        time.sleep(1 / metadata.fps)

    print(f"{metadata.width = }")
    print(f"{metadata.height = }")
    print(f"{metadata.fps = }")

    # vid = Vid(metadata.fps, metadata.width, metadata.height, frames)
    # vid.save_to_file("output.vid")