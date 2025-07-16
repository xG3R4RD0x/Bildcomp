# import unittest
# import numpy as np
# from pipeline.stages.entropie.huffmann_codierung import HuffmannCoding
# from pipeline.stages.quantization.quantization_stage import QuantizationStage
# from pipeline.stages.decorrelation.strategy.transformation_strategy import BlockDCT

# class TestHuffmannCoding(unittest.TestCase):
#     def setUp(self):
#         self.huffman = HuffmannCoding()
#         # Carga el primer frame del video de prueba
#         self.input_data = np.fromfile(
#             "test_videos/Sign_Irene_352x288.yuv", dtype=np.uint8
#         )
#         self.input_height = 288
#         self.input_width = 352
#         self.frame_size = self.input_width * self.input_height * 3 // 2
#         self.first_frame = self.input_data[:self.frame_size]

#     def test_identity(self):
#         # Datos simples, todos los bytes diferentes
#         data = bytes(range(256))
#         encoded = self.huffman.process(data)
#         decoded = self.huffman.process(encoded, decode=True)
#         self.assertEqual(data, decoded)

#     def test_repeated_bytes(self):
#         # Data with repetitions
#         data = b'A' * 1000 + b'B' * 500 + b'C' * 100
#         encoded = self.huffman.process(data)
#         decoded = self.huffman.process(encoded, decode=True)
#         self.assertEqual(data, decoded)

#     def test_empty(self):
#         data = b''
#         encoded = self.huffman.process(data)
#         decoded = self.huffman.process(encoded, decode=True)
#         self.assertEqual(data, decoded)

#     def test_random(self):
#         import os
#         data = os.urandom(1024)
#         encoded = self.huffman.process(data)
#         decoded = self.huffman.process(encoded, decode=True)
#         self.assertEqual(data, decoded)

#     def test_huffman_frame(self):
#         """
#         Encodes and decodes the first frame of the video using DCT, quantization, and Huffman.
#         """
#         y_size = self.input_width * self.input_height
#         y = self.first_frame[:y_size].astype(np.float64).reshape((self.input_height, self.input_width))
#         # DCT
#         dct = BlockDCT(width=self.input_width, height=self.input_height, block_size=8)
#         transformed = dct.transform(y)
#         # Cuantización
#         quant = QuantizationStage()
#         levels = 128
#         quantized, data_min, step = quant.process(transformed.flatten(), levels=levels)
#         # Codificación Huffman
#         encoded = self.huffman.process(quantized.tobytes())
#         # Decodificación Huffman
#         decoded_bytes = self.huffman.process(encoded, decode=True)
#         decoded = np.frombuffer(decoded_bytes, dtype=quantized.dtype)
#         # De-cuantización
#         dequantized = quant.dequantize(decoded, data_min, step)
#         dequantized = np.array(dequantized).reshape(transformed.shape)
#         # Comprobación de forma y error
#         self.assertEqual(transformed.shape, dequantized.shape)
#         mae = np.mean(np.abs(transformed - dequantized))
#         print(f"MAE Huffman+DCT+Quant: {mae}, Step: {step}")

# if __name__ == '__main__':
#     unittest.main()
