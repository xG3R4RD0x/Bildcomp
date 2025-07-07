# from pipeline.interfaces.base_stage import CompressionStage
# from pipeline.interfaces.bitwriter import BitWriter, BitReader
# import math
# from typing import Dict, Tuple


# class HuffmannCoding(CompressionStage):
#     def name(self) -> str:
#         return "Huffman Coding Stage"

#     def occurence_distribution(self, data: list[int]) -> Dict[int, int]:
#         occurences = dict()
#         for x in data:
#             if x not in occurences:
#                 occurences[x] = 1
#             else:
#                 occurences[x] += 1
#         return occurences

#     def huffman_coding_length_limited(self, sorted_occurenes: list[int], max_bits: int) -> list[int]:
#         initial_packages: list[Tuple[int, list[int]]] = []
#         for i, occ in enumerate(sorted_occurenes):
#             initial_packages.append((occ, [i]))
#         initial_packages = sorted(initial_packages, key=lambda p: (p[0], -len(p[1])))
#         packages = initial_packages.copy()
#         for _ in range(max_bits - 1):
#             new_packages = []
#             for i in range(len(packages) // 2):
#                 p0 = packages[i * 2 + 0]
#                 p1 = packages[i * 2 + 1]
#                 items = []
#                 items.extend(p0[1])
#                 items.extend(p1[1])
#                 p0p1 = (p0[0] + p1[0], items)
#                 new_packages.append(p0p1)
#             packages = new_packages + initial_packages
#             packages = sorted(packages, key=lambda p: (p[0], -len(p[1])))
#         num_packages = 2 * len(sorted_occurenes) - 2
#         symbols = [p[1] for p in packages[:num_packages]]
#         symbols_flattened = [s for syms in symbols for s in syms]
#         bit_lengths = []
#         for i in range(len(sorted_occurenes)):
#             bit_lengths.append(symbols_flattened.count(i))
#         return bit_lengths

#     def huffman_codes_from_bit_lengths(self, bit_lengths: list[int]) -> list[int]:
#         MAX_BITS = max(bit_lengths) if bit_lengths else 0
#         bl_count = [bit_lengths.count(i) for i in range(0, MAX_BITS)]
#         next_code = [0]
#         if bl_count:
#             bl_count[0] = 0
#         code = 0
#         for bits in range(1, MAX_BITS + 1):
#             code = (code + bl_count[bits - 1]) << 1
#             next_code.append(code)
#         codes = []
#         for length in bit_lengths:
#             if length != 0:
#                 codes.append(next_code[length])
#                 next_code[length] += 1
#             else:
#                 codes.append(0)
#         return codes

#     def process(self, data: bytes, decode: bool = False) -> bytes:
#         if not decode:
#             # ENCODING
#             data_list = list(data)
#             occurences = self.occurence_distribution(data_list)
#             symbols = sorted(occurences.keys())
#             symbol_counts = [occurences[s] for s in symbols]
#             # Build Huffman tree (length-limited)
#             bitlengths = self.huffman_coding_length_limited(symbol_counts, 15)
#             # Map bitlengths to all possible byte values (0-255)
#             bitlengths_all = [0]*256
#             for idx, sym in enumerate(symbols):
#                 bitlengths_all[sym] = bitlengths[idx]
#             codes = self.huffman_codes_from_bit_lengths(bitlengths_all)
#             # Write header: store bitlengths for all 256 symbols (1 byte each)
#             header = bytes(bitlengths_all)
#             # Encode data
#             writer = BitWriter()
#             for b in data_list:
#                 length = bitlengths_all[b]
#                 code = codes[b]
#                 writer.write(code, length)
#             writer.flush()
#             return header + writer.output_bytes
#         else:
#             # DECODING
#             bitlengths_all = list(data[:256])
#             codes = self.huffman_codes_from_bit_lengths(bitlengths_all)
#             # Build decoding table: (length, code) -> symbol
#             decode_table = {}
#             for symbol, length in enumerate(bitlengths_all):
#                 if length != 0:
#                     code = codes[symbol]
#                     decode_table[(length, code)] = symbol
#             # Prepare to decode
#             reader = BitReader(data[256:])
#             result = []
#             # To know how many symbols to decode, we need to store the original length somewhere.
#             # For demo, decode until EOF (may need to adapt for real use)
#             try:
#                 while True:
#                     # Try all possible lengths (shortest first)
#                     for l in range(1, max(bitlengths_all)+1):
#                         code = reader.peek(l)
#                         key = (l, code)
#                         if key in decode_table:
#                             reader.read(l)
#                             result.append(decode_table[key])
#                             break
#                     else:
#                         # No valid code found
#                         break
#             except EOFError:
#                 pass
#             return bytes(result)