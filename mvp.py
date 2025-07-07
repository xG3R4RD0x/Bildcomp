import math
import struct
import sys
from typing import Dict, Self, Tuple
from os.path import basename

try:
    from numba import njit
except ModuleNotFoundError:
    print("Info: You do not seem to have numba installed on your system. This program will work without it, but it may be significantly slower.")

    def njit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

##### configuration settings #####

BLOCK_SIZE = (8, 8)

SQRT2_REPRO = 1/math.sqrt(2)
SQRT2_OVER_BLOCK_WIDTH = math.sqrt(2/BLOCK_SIZE[0])
SQRT2_OVER_BLOCK_HEIGHT = math.sqrt(2/BLOCK_SIZE[1])

CODEWORD_MAX_BITS = 15
CODEWORD_LENGTH_BITS = int.bit_length(CODEWORD_MAX_BITS)

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

def parse_raw_metadata_from_filename(path: str) -> RawYUVMetadata | None:
    dimensions_str = None
    fps_str = None

    filename = basename(path)
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
    if fps_str is not None:
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

    def calculate_psnr(a: Self, b: Self) -> float:
        assert(a.width == b.width)
        assert(a.height == b.height)
        assert(len(a.y) == len(b.y))
        assert(len(a.u) == len(b.u))
        assert(len(a.v) == len(b.v))

        squared_error = 0
        for ay, by in zip(a.y, b.y):
            squared_error += (ay - by) * (ay - by)
        for au, bu in zip(a.u, b.u):
            squared_error += (au - bu) * (au - bu)
        for av, bv in zip(a.v, b.v):
            squared_error += (av - bv) * (av - bv)
        num_elements = len(a.y) + len(a.u) + len(a.v)

        psnr_max2 = 255 * 255
        psnr_mse = squared_error / num_elements
        if (psnr_mse == 0):
            return math.inf
        psnr = 10 * math.log10(psnr_max2 / psnr_mse)
        return psnr


class Vid:
    fps: int
    frame_width: int
    frame_height: int

    quantization_y_interval: float
    quantization_u_interval: float
    quantization_v_interval: float

    frames: list[YUVImage]

    def __init__(self, fps: int, frame_width: int, frame_height: int, frames: list[YUVImage]) -> Self:
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.quantization_y_interval = 10.0
        self.quantization_u_interval = 10.0
        self.quantization_v_interval = 10.0

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

        self.quantization_y_interval = struct.unpack('<f', content[10:14])
        self.quantization_u_interval = struct.unpack('<f', content[14:18])
        self.quantization_v_interval = struct.unpack('<f', content[18:22])

        bitreader = BitReader(content[22:])

        def read_frame_data(bitreader: BitReader, width: int, height: int) -> list[int]:
            quantization_interval = bitreader.read_float()
            if quantization_interval is None:
                return None

            quantization_levels = bitreader.read(32)
            bitlengths = []
            for i in range(quantization_levels):
                bitlen = bitreader.read(CODEWORD_LENGTH_BITS)
                bitlengths.append(bitlen)
            huffman_codes = huffman_codes_from_bit_lengths(bitlengths)
            sorted_bitlength_set = sorted(set(bitlengths))

            dequantized_data = []
            for i in range(width * height):
                # print(f"reading y value: {i} of {width * height}")
                quantized_value = bitreader.read_huffman(sorted_bitlength_set, bitlengths, huffman_codes)
                dequantized_value = quantized_value - (quantization_levels // 2)
                dequantized_value *= quantization_interval
                dequantized_data.append(dequantized_value)

            data = inverse_discrete_cosine_transform_2d(width, height, width, dequantized_data)
            return bytes(data)

        # print(self.frame_width, self.frame_height, self.frame_width * self.frame_height)
        while y_data := read_frame_data(bitreader, self.frame_width, self.frame_height):
            u_data = read_frame_data(bitreader, self.frame_width // 2, self.frame_height // 2)
            assert u_data is not None
            v_data = read_frame_data(bitreader, self.frame_width // 2, self.frame_height // 2)
            assert v_data is not None

            frame = YUVImage(self.frame_width, self.frame_height, y_data, u_data, v_data)
            # frame.save_as_tga('output.tga')

            self.frames.append(frame)
            print(f"Decoding frame: {len(self.frames)}\r", end='')

        print(f"Decoding frame: {len(self.frames)}")
        return self

    def save_to_file(self, filename: str):
        frames_bitwriter = BitWriter()
        def write_frame_data(quantization_interval: float, length_code_pairs: list[Tuple[int, int]], data: list[int]):
            for byte in struct.pack('<f', quantization_interval):
                frames_bitwriter.write(byte, 8)

            frames_bitwriter.write(len(length_code_pairs), 32)
            for bitlen, _ in length_code_pairs:
                assert bitlen <= CODEWORD_MAX_BITS
                frames_bitwriter.write(bitlen, CODEWORD_LENGTH_BITS)

            for i, x in enumerate(data):
                length, code = length_code_pairs[x]
                frames_bitwriter.write(code, length)

        for i, frame in enumerate(self.frames):
            print(f"Writing frame: {i}\r", end="")
            components = [
                (frame.y, self.quantization_y_interval, self.frame_width, self.frame_height),
                (frame.u, self.quantization_u_interval, self.frame_width // 2, self.frame_height // 2),
                (frame.v, self.quantization_v_interval, self.frame_width // 2, self.frame_height // 2),
            ]
            for (c, quantization_interval, width, height) in components:
                frame_quantized = [0] * width * height
                frame_reconstructed = [0] * width * height

                assert width % BLOCK_SIZE[0] == 0
                assert height % BLOCK_SIZE[1] == 0
                blocks_per_row = width // BLOCK_SIZE[0]
                blocks_per_column = height // BLOCK_SIZE[1]

                for block_y in range(blocks_per_column):
                    for block_x in range(blocks_per_row):
                        byte_offset = (block_y * blocks_per_row * BLOCK_SIZE[0] * BLOCK_SIZE[1]) + (block_x * BLOCK_SIZE[0])

                        # decorrelation
                        # TODO: implement prediction
                        c_transformed = discrete_cosine_transform_block(width, c[byte_offset:])
                        c_quantized = [round(x / quantization_interval) for x in c_transformed]

                        # reconstruct frame for prediction
                        c_reconstructed = [x * quantization_interval for x in c_quantized]
                        c_inverse_transformed = inverse_discrete_cosine_transform_2d(BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[1], c_reconstructed)

                        # store data in frame
                        for y in range(BLOCK_SIZE[1]):
                            for x in range(BLOCK_SIZE[0]):
                                src_offset = (y * BLOCK_SIZE[0]) + x
                                dest_offset = byte_offset + (y * width) + x
                                frame_quantized[dest_offset] = c_quantized[src_offset]

                                # for prediction
                                frame_reconstructed[dest_offset] = c_inverse_transformed[src_offset]

                # quantization
                quantized_max = max(abs(x) for x in frame_quantized)
                quantization_levels = quantized_max * 2 + 2
                frame_quantized = [x + (quantization_levels // 2) for x in frame_quantized]

                # probability modelling
                occurences = occurence_distribution(frame_quantized)
                symbols = []
                symbol_counts = []
                for k, v in occurences.items():
                    symbols.append(k)
                    symbol_counts.append(v)

                # entropy coding
                bitlengths = huffman_coding_length_limited(symbol_counts, CODEWORD_MAX_BITS)

                bitlengths_all = []
                # TODO: symbols is sorted, so don't search
                for i in range(quantization_levels):
                    bitlen = 0
                    if i in symbols:
                        index = symbols.index(i)
                        bitlen = bitlengths[index]
                    bitlengths_all.append(bitlen)

                codes = huffman_codes_from_bit_lengths(bitlengths_all)
                length_code_pairs = list(zip(bitlengths_all, codes))
                assert(len(length_code_pairs) == quantization_levels)

                write_frame_data(quantization_interval, length_code_pairs, frame_quantized)

        print(f"Written all {len(self.frames)} frames")
        frames_bitwriter.flush()

        with open(filename, "wb") as f:
            f.write(b".VID")
            f.write(int.to_bytes(0, 1, 'little'))

            f.write(int.to_bytes(self.fps, 1, 'little'))
            f.write(int.to_bytes(self.frame_width, 2, 'little'))
            f.write(int.to_bytes(self.frame_height, 2, 'little'))

            f.write(struct.pack('<f', self.quantization_y_interval))
            f.write(struct.pack('<f', self.quantization_u_interval))
            f.write(struct.pack('<f', self.quantization_v_interval))

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
def discrete_cosine_transform_block(stride: int, data: list[int]) -> list[float]:
    transformed = [0] * BLOCK_SIZE[0] * BLOCK_SIZE[1]
    block = [0] * BLOCK_SIZE[0] * BLOCK_SIZE[1]

    # TODO: assumes width and height multiple of 8, fix that
    # transform horizontally
    N = BLOCK_SIZE[0]
    for y in range(BLOCK_SIZE[1]):
        for k in range(N):
            C0 = SQRT2_REPRO if k == 0 else 1

            summed_xs = 0
            for x in range(BLOCK_SIZE[0]):
                offset = (y * stride) + x
                n = x
                summed_xs += data[offset] * math.cos((2 * n + 1) * math.pi * k / (2 * N))

            X = C0 * SQRT2_OVER_BLOCK_WIDTH * summed_xs
            block[y * BLOCK_SIZE[0] + k] = X

    # transform vertically
    N = BLOCK_SIZE[1]
    for x in range(BLOCK_SIZE[0]):
        for k in range(N):
            C0 = SQRT2_REPRO if k == 0 else 1

            summed_xs = 0
            for y in range(BLOCK_SIZE[1]):
                offset = (y * BLOCK_SIZE[0]) + x
                n = y
                summed_xs += block[offset] * math.cos((2 * n + 1) * math.pi * k / (2 * N))

            X = C0 * SQRT2_OVER_BLOCK_HEIGHT * summed_xs
            offset = (k * BLOCK_SIZE[0]) + x # NOTE: we are intentionally using k and width instead of stride, because this is in terms of the output coefficients
            transformed[offset] = X

    return transformed

@njit
def inverse_discrete_cosine_transform_2d(width: int, height: int, stride: int, data: list[float]) -> list[int]:
    reconstructed = [0] * width * height

    block_width = 8
    block_height = 8
    blocks_per_row = width // block_width
    block = [0] * (block_width * block_height)

    num_blocks = (width * height) // (block_width * block_height)
    for block_index in range(num_blocks):
        yoff = block_height * (block_index // blocks_per_row)
        xoff = block_width * (block_index % blocks_per_row)

        # inverse transform vertically
        N = block_height
        for x in range(block_width):
            for n in range(N):

                summed_xs = 0
                for y in range(block_height):
                    offset = ((y + yoff) * stride) + (x + xoff)
                    k = y

                    C0 = SQRT2_REPRO if k == 0 else 1
                    summed_xs += C0 * data[offset] * math.cos((2 * n + 1) * k * math.pi / (2 * N))

                X = SQRT2_OVER_BLOCK_HEIGHT * summed_xs
                offset = (n * block_width) + x
                block[offset] = X

        # inverse transform horizontally
        N = block_width
        for y in range(block_height):
            for n in range(N):

                summed_xs = 0
                for x in range(block_width):
                    offset = (y * block_height) + x
                    k = x

                    C0 = SQRT2_REPRO if k == 0 else 1
                    summed_xs += C0 * block[offset] * math.cos((2 * n + 1) * k * math.pi / (2 * N))

                X = SQRT2_OVER_BLOCK_WIDTH * summed_xs
                X = max(0, min(int(round(X)), 255)) # clamp the output, so we have valid byte values for each component
                offset = ((y + yoff) * width) + (n + xoff)
                reconstructed[offset] = X

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

        for j in range(len(packages) // 2):
            p0: Tuple[int, list[int]] = packages[j * 2 + 0] # every even indexed item
            p1: Tuple[int, list[int]] = packages[j * 2 + 1] # every  odd indexed item

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
        assert symbols_flattened.count(i) <= max_bits
        bit_lengths.append(symbols_flattened.count(i))

    return bit_lengths

# @njit
def huffman_codes_from_bit_lengths(bit_lengths: list[int]) -> list[int]:
    # https://www.rfc-editor.org/rfc/rfc1951.html#section-3.2.2
    MAX_BITS = max(bit_lengths)

    bl_count = []
    for i in range(0, max(bit_lengths) + 1):
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

def print_usage_and_exit():
    args = sys.argv
    exit(f"""Usage: python3 {args[0]} <subcommand> [options]

  Available Subcommands:
    compress   <input.yuv> <output.vid>   Compress the raw video data from the specified input.yuv file and write the resulting video to the specified output.vid file. Try to extract the metadata from the filename itself. 
    decompress <input.vid> <output.yuv>   Read the compressed video data from the specified input.vid file. Decompress the video and write the resulting raw video data to the specified output.yuv file.

  Available Options:
    --quantization-interval <float>       Only available for the compress subcommand. Specify the desired quantization interval used for compression. We recommend a value between 1.0 and 15.0.
    --help                                Show this help and quit.
""")

##### main logic #####
if __name__ == '__main__':
    args = sys.argv

    subcommand = None
    input_file = None
    output_file = None
    quantization_interval = None

    # parse command line arguments
    argIndex = 1 # index 0 is the program itself
    while argIndex < len(args):
        if args[argIndex].startswith('--'):
            if args[argIndex] == '--help':
                print_usage_and_exit()
            if args[argIndex] == '--quantization-interval':
                if (argIndex + 1) >= len(args):
                    # error: missing argument for option --quantization-interval
                    print_usage_and_exit()

                try:
                    quantization_interval = float(args[argIndex + 1])
                    argIndex += 1
                except ValueError:
                    exit(f"Invalid argument for --quantization-interval. Expected a number between 1.0 and 15.0 got '{args[argIndex + 1]}'")
            else:
                exit(f"Unexpected option: {args[argIndex]}")
        elif subcommand is None:
            if args[argIndex] == "compress" or args[argIndex] == "decompress":
                subcommand = args[argIndex]
            else:
                print_usage_and_exit()
        elif input_file is None:
            input_file = args[argIndex]
        elif output_file is None:
            output_file = args[argIndex]
        else:
            exit(f"Unexpected argument: {args[argIndex]}")

        argIndex += 1

    if subcommand is None or input_file is None or output_file is None:
        print_usage_and_exit()


    # execute the program with the given subcommand and options
    if subcommand == "compress":
        content = None
        with open(input_file, 'rb') as f:
            content = f.read()

        metadata = parse_raw_metadata_from_filename(input_file)

        ys_size = metadata.width * metadata.height * 1
        us_size = metadata.width * metadata.height // 4
        vs_size = metadata.width * metadata.height // 4

        frame_size = ys_size + us_size + vs_size

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

        print("")

        vid = Vid(metadata.fps, metadata.width, metadata.height, frames)
        if quantization_interval is not None:
            vid.quantization_y_interval = quantization_interval
            vid.quantization_u_interval = quantization_interval
            vid.quantization_v_interval = quantization_interval
        vid.save_to_file(output_file)
    elif subcommand == "decompress":
        vid = Vid.read_from_file(input_file)

        with open(output_file, 'wb') as f:
            for frame in vid.frames:
                f.write(frame.y)
                f.write(frame.u)
                f.write(frame.v)
    else:
        assert False, "Invalid subcommand. This should never happen."
