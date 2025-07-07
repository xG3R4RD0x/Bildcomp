import unittest
import numpy as np
from pipeline.stages.quantization.quantization_stage import quantize, dequantize
from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage,separate_yuv_compression
from pipeline.stages.decorrelation.strategy.transformation_strategy import BlockDCT as bd
from pipeline.stages.decorrelation.strategy.transformation_strategy import precompute_cosines, dct_block, idct_block

class TestQuantizationStage(unittest.TestCase):
    
    def setUp(self):

        # load sample video
        self.input_data = np.fromfile(
            "test_videos/Sign_Irene_352x288.yuv", dtype=np.uint8
        )
        self.input_height = 288
        self.input_width = 352

    def test_dct_quantization_and_reconstruction(self):
        """
        Verifica que un frame pasado por DCT, cuantizado y de-cuantizado se reconstruye correctamente.
        """
        frame_size = self.input_width * self.input_height * 3 // 2
        first_frame = self.input_data[:frame_size]

        video = separate_yuv_compression(first_frame, self.input_width, self.input_height)
        original_y = video[0].astype(np.float64)
        block_size = 8
        cosines = precompute_cosines(block_size)
        transformed = dct_block(original_y, cosines, block_size)

        
        levels = 128
        quantized, data_min, step = quantize(transformed, levels=levels)
        dequantized = dequantize(quantized, data_min, step)
        #asegurar la forma
        dequantized = np.array(dequantized).reshape(transformed.shape)

        self.assertEqual(transformed.shape, dequantized.shape)

        mae = np.mean(np.abs(transformed - dequantized))
        print(f"MAE: {mae}, Step: {step}")
        self.assertLess(mae, step)

        self.assertTrue(np.all(quantized >= 0))
        self.assertTrue(np.all(quantized < levels))

        self.assertFalse(np.isnan(dequantized).any())
        self.assertFalse(np.isinf(dequantized).any())

    def test_full_video_dct_quantization_and_reconstruction(self):
        """
        Verifica que todos los frames del video, tras DCT + cuantización + de-cuantización,
        se reconstruyan correctamente con error limitado.
        """
        width, height = self.input_width, self.input_height
        frame_size = width * height * 3 // 2
        num_frames = len(self.input_data) // frame_size
        block_size = 8

        dct = bd(width=width, height=height, block_size=block_size)
        cosines = precompute_cosines(block_size)
        levels = 128
        

        for frame_index in range(num_frames):
            offset = frame_index * frame_size
            frame = self.input_data[offset : offset + frame_size]
            video = separate_yuv_compression(frame, width, height)
            original_y = video[0].astype(np.float64)

            transformed = dct_block(original_y, cosines, block_size)
            quantized, data_min, step = quantize(transformed, levels=levels)
            dequantized = dequantize(quantized, data_min, step).reshape(transformed.shape)

            # 1. Forma coherente
            self.assertEqual(transformed.shape, dequantized.shape)

            # 2. MAE debe ser menor al step
            mae = np.mean(np.abs(transformed - dequantized))
            print(f"Frame {frame_index}: MAE = {mae}, Step = {step}")
            self.assertLess(mae, step)

            # 3. Rango de cuantización correcto
            self.assertTrue(np.all(quantized >= 0))
            self.assertTrue(np.all(quantized < levels))

            # 4. No hay NaNs ni infs
            self.assertFalse(np.isnan(dequantized).any())
            self.assertFalse(np.isinf(dequantized).any())
