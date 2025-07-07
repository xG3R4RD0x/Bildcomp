
import numpy as np
import heapq
from collections import Counter, namedtuple

class HuffmanNode(namedtuple('HuffmanNode', ['symbol', 'freq', 'left', 'right'])):
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbols):
    # symbols: 1D np.ndarray or list of ints
    freq = Counter(symbols)
    heap = [HuffmanNode(sym, f, None, None) for sym, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # Only one symbol: assign code '0'
        node = heap[0]
        return HuffmanNode(None, node.freq, node, None)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
    return heap[0]

def build_huffman_table(tree):
    # Returns {symbol: bitstring}
    table = {}
    def _walk(node, code):
        if node.left is None and node.right is None:
            table[node.symbol] = code or '0'
            return
        if node.left:
            _walk(node.left, code + '0')
        if node.right:
            _walk(node.right, code + '1')
    _walk(tree, '')
    return table

def encode_symbols(symbols, table):
    # symbols: 1D np.ndarray or list of ints
    # table: {symbol: bitstring}
    bitstring = ''.join(table[sym] for sym in symbols)
    # Pack bits into bytes
    n = 8
    pad = (n - len(bitstring) % n) % n
    bitstring += '0' * pad
    b = bytearray()
    for i in range(0, len(bitstring), n):
        b.append(int(bitstring[i:i+n], 2))
    return bytes(b), pad

def decode_symbols(encoded_bytes, table, num_symbols, pad):
    # table: {symbol: bitstring}
    # Reverse table
    rev_table = {v: k for k, v in table.items()}
    # Convert bytes to bitstring
    bitstring = ''.join(f'{byte:08b}' for byte in encoded_bytes)
    if pad:
        bitstring = bitstring[:-pad]
    # Decode
    out = []
    code = ''
    for bit in bitstring:
        code += bit
        if code in rev_table:
            out.append(rev_table[code])
            code = ''
            if len(out) == num_symbols:
                break
    return np.array(out)

def huffman_encode_frame(quantized_blocks):
    """
    compressed_blocks: list of np.ndarray, one per block (shape: (block_size, block_size))
    Returns:
        encoded_blocks: list of bytes, one per block
        huff_table: {symbol: bitstring}
        pads: list of int, number of padding bits per block
        shape: (n_blocks_y, n_blocks_x, block_size)
    """
    # Recopila todos los valores de todos los bloques (solo los datos, no metadatos)
    all_values = []
    n_blocks_y = len(quantized_blocks)
    n_blocks_x = len(quantized_blocks[0])
    block_size = quantized_blocks[0][0].shape[0]
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = quantized_blocks[by][bx] # debe ser np.ndarray (block_size, block_size)
            all_values.extend(block.flatten())
    all_values = np.array(all_values)
    # Construye el árbol y la tabla de Huffman usando todos los valores
    tree = build_huffman_tree(all_values)
    table = build_huffman_table(tree)
    # Codifica cada bloque usando la tabla
    encoded_blocks = []
    pads = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = quantized_blocks[by][bx].flatten()
            encoded, pad = encode_symbols(block, table)
            encoded_blocks.append(encoded)
            pads.append(pad)
    # El árbol (o la tabla) debe ir como metadata del frame
    return encoded_blocks, table, pads, (n_blocks_y, n_blocks_x, block_size)

def huffman_decode_frame(encoded_blocks, table, pads, shape):
    """
    encoded_blocks: list of bytes, one por bloque
    table: {symbol: bitstring}
    pads: list of int, uno por bloque
    shape: (n_blocks_y, n_blocks_x, block_size)
    Returns: list de np.ndarray, uno por bloque (block_size, block_size)
    """
    n_blocks_y, n_blocks_x, block_size = shape
    blocks = []
    idx = 0
    for by in range(n_blocks_y):
        row = []
        for bx in range(n_blocks_x):
            num_symbols = block_size * block_size
            decoded = decode_symbols(encoded_blocks[idx], table, num_symbols, pads[idx])
            row.append(decoded.reshape((block_size, block_size)))
            idx += 1
        blocks.append(row)
    return blocks

# Example usage:
# encoded_blocks, table, pads, shape = huffman_encode_frame(quantized_blocks)
# decoded_blocks = huffman_decode_frame(encoded_blocks, table, pads, shape)
# Huffman coding stage for entropy coding after quantization
# Generates a Huffman table per frame and encodes quantized values blockwise
