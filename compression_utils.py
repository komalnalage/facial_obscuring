import heapq
from typing import Counter
import cv2
import numpy as np
import zlib
import pickle

def rle_compress(image):
    flat = image.flatten()
    compressed = []
    prev = flat[0]
    count = 1
    for i in flat[1:]:
        if i == prev:
            count += 1
        else:
            compressed.append((prev, count))
            prev = i
            count = 1
    compressed.append((prev, count))
    return compressed


# ---------- Huffman Compression ----------
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0] if heap else None

def build_codes(root):
    codes = {}
    def _generate_codes(node, code=""):
        if node:
            if node.symbol is not None:
                codes[node.symbol] = code
            _generate_codes(node.left, code + "0")
            _generate_codes(node.right, code + "1")
    _generate_codes(root)
    return codes

def huffman_compress(image):
    flat = image.flatten()
    freq_dict = Counter(flat)
    root = build_huffman_tree(freq_dict)
    codes = build_codes(root)

    encoded_data = "".join([codes[pixel] for pixel in flat])

    extra_padding = 8 - len(encoded_data) % 8
    encoded_data += "0" * extra_padding

    padded_info = "{0:08b}".format(extra_padding)
    encoded_data = padded_info + encoded_data

    b = bytearray()
    for i in range(0, len(encoded_data), 8):
        byte = encoded_data[i:i+8]
        b.append(int(byte, 2))


    return {
        "compressed_data": bytes(b),
        "codes": codes,
        "shape": image.shape,
        "padding": extra_padding
    }

def calculate_compression_ratio(original_img, compressed):
    original_size = original_img.nbytes if isinstance(original_img, np.ndarray) else len(pickle.dumps(original_img))

    if isinstance(compressed, (bytes, bytearray)):
        compressed_size = len(compressed)
    elif isinstance(compressed, list):
        compressed_size = len(compressed) * 2  
    elif isinstance(compressed, dict) and "compressed_data" in compressed:
        compressed_size = len(compressed["compressed_data"]) + len(str(compressed["codes"]).encode())
    else:
        compressed_size = len(pickle.dumps(compressed))

    # Calculate percentage reduction
    ratio = (1 - (compressed_size / original_size)) * 100
    return ratio
