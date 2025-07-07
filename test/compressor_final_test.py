import struct
import unittest
import os
from PIL import Image

from pipeline.compression.compressor_final import CompressorFinal, compress_block_prediction_dct_and_quant, compress_frame_test, decompress_block_prediction_dct_and_quant, decompress_frame_test
from pipeline.stages.decorrelation.decorrelation_stage import separate_yuv_compression
from pipeline.stages.bitwriter.bitwritter import BitWriter
import numpy as np

from pipeline.stages.decorrelation.strategy.transformation_strategy import precompute_cosines

class CompressorFinalTest(unittest.TestCase):
   
    def setUp(self):
        self.video_path = "test_videos/Sign_Irene_352x288.yuv"
        self.video_path_compressed = "test_videos/compressed/Sign_Irene_352x288.finalcomp"
        self.video_path_decompressed = "test_videos/decompressed/Sign_Irene_352x288.yuv"
        self.width = 352
        self.height = 288
        
    def test_final_compressor(self):
        """
        Test the final compressor functionality
        """
        compressor = CompressorFinal()

        # Compress the video
        compressor.compress_video(
            video_path=self.video_path,
            output_path="test_videos/compressed/",
            height=self.height,
            width=self.width,
            block_size=8,
            levels=32,
        )

        # Check if the compressed file exists
        self.assertTrue(os.path.exists(self.video_path_compressed))

        # Read the header from the compressed file
        with open(self.video_path_compressed, "rb") as f:
            header = f.read(8)
            width, height, num_frames, block_size, levels = struct.unpack("<HHHBB", header)
        # Assert values match what was passed
        self.assertEqual(width, self.width)
        self.assertEqual(height, self.height)
        self.assertEqual(block_size, 8)
        

        # Decompress the video
        compressor.decompress_video(
            compressed_path=self.video_path_compressed,
            output_path="test_videos/decompressed/"
        )

        # Check if the decompressed file exists
        decompressed_path = os.path.join(
            "test_videos/decompressed/",
            os.path.splitext(os.path.basename(self.video_path_compressed))[0] + ".yuv"
        )
        self.assertTrue(os.path.exists(decompressed_path))

        # Compare original and decompressed file sizes (should be equal for YUV 4:2:0)
        orig_size = os.path.getsize(self.video_path)
        decomp_size = os.path.getsize(decompressed_path)
        self.assertEqual(orig_size, decomp_size)

        # Optionally, compare first N bytes for quick sanity check (not strict equality)
        with open(self.video_path, "rb") as f1, open(decompressed_path, "rb") as f2:
            orig_bytes = f1.read(1048576)
            decomp_bytes = f2.read(1048576)
        # At least not all zeros and same length
        self.assertEqual(len(orig_bytes), len(decomp_bytes))
        self.assertTrue(any(b != 0 for b in decomp_bytes))
        
    def test_compress_decompress_frame_test_on_first_frame(self):
        """
        Test compress_frame_test y decompress_frame_test en el canal Y del primer frame
        """

        # Lee el archivo completo como array 1D
        input_data = np.fromfile(self.video_path, dtype=np.uint8)
        frame_size = self.width * self.height * 3 // 2
        first_frame = input_data[:frame_size]
        # Separa los canales YUV
        yuv_dict = separate_yuv_compression(first_frame, self.width, self.height)
        block_size = 8
        cosines = precompute_cosines(block_size)

        # Definir parámetros para cada canal
        channels = [
            (0, self.width, self.height, 'Y'),
            (1, self.width // 2, self.height // 2, 'U'),
            (2, self.width // 2, self.height // 2, 'V'),
        ]

        for idx, w, h, label in channels:
            channel = yuv_dict[idx]
            padded_h = ((h + block_size - 1) // block_size) * block_size
            padded_w = ((w + block_size - 1) // block_size) * block_size

            # Comprime el canal usando compress_frame_test
            compressed = compress_frame_test(
                frame_data=channel.tobytes(),
                width=w,
                height=h,
                padded_w=padded_w,
                padded_h=padded_h,
                block_size=block_size,
                levels=128,
                cosines=cosines
            )

            # Descomprime el canal usando decompress_frame_test
            decompressed = decompress_frame_test(
                compressed,
                w,
                h,
                block_size,
                128,
                cosines
            )

            # Verifica que la salida tenga la forma esperada
            self.assertEqual(decompressed.shape, (h, w), msg=f"Shape mismatch in channel {label}")
            self.assertTrue(np.issubdtype(decompressed.dtype, np.integer), msg=f"Dtype not integer in channel {label}")
            # El frame descomprimido no debe ser todo ceros
            self.assertTrue(np.any(decompressed), msg=f"All zeros in channel {label}")

            # Calcular diferencia absoluta media y máxima entre original y reconstruido
            abs_diff = np.abs(channel.astype(np.float32) - decompressed.astype(np.float32))
            mean_diff = np.mean(abs_diff)
            max_diff = np.max(abs_diff)
            print(f"[{label}] Mean abs diff: {mean_diff:.2f}, Max abs diff: {max_diff:.2f}")
            # El error medio debe ser razonable para compresión con pérdidas (ajustar umbral si necesario)
            self.assertLess(mean_diff, 30, msg=f"Mean diff too high in channel {label}")

            # Calcular PSNR entre bloque original y reconstruido
            mse = np.mean((channel.astype(np.float32) - decompressed.astype(np.float32)) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                PIXEL_MAX = 255.0
                psnr = 10 * np.log10((PIXEL_MAX ** 2) / mse)
            print(f"[{label}] PSNR: {psnr:.2f} dB")
            # El PSNR debe ser razonable para compresión con pérdidas (ajustar umbral si necesario)
            self.assertGreater(psnr, 20, msg=f"PSNR too low in channel {label}")
            
            # Guarda el frame YUV original como imagen RGB para inspección visual
            frame_yuv = bytearray()
            frame_yuv.extend(yuv_dict[0].astype(np.uint8).tobytes())
            frame_yuv.extend(yuv_dict[1].astype(np.uint8).tobytes())
            frame_yuv.extend(yuv_dict[2].astype(np.uint8).tobytes())
            self.yuv420_to_rgb_and_save(frame_yuv, self.width, self.height, "test_videos/decompressed/test_frame.png")
    def test_compress_decompress_frame_test_on_all_frames(self):
        """
        Test compress_frame_test y decompress_frame_test en todos los frames
        """

        # Lee el archivo completo como array 1D
        input_data = np.fromfile(self.video_path, dtype=np.uint8)
        frame_size = self.width * self.height * 3 // 2
        num_frames = len(input_data) // frame_size
        block_size = 8
        cosines = precompute_cosines(block_size)

        for frame_idx in range(num_frames):
            frame_start = frame_idx * frame_size
            frame_end = frame_start + frame_size
            frame_bytes = input_data[frame_start:frame_end]
            yuv_dict = separate_yuv_compression(frame_bytes, self.width, self.height)
            channels = [
                (0, self.width, self.height, 'Y'),
                (1, self.width // 2, self.height // 2, 'U'),
                (2, self.width // 2, self.height // 2, 'V'),
            ]
            for idx, w, h, label in channels:
                channel = yuv_dict[idx]
                padded_h = ((h + block_size - 1) // block_size) * block_size
                padded_w = ((w + block_size - 1) // block_size) * block_size

                # Comprime el canal usando compress_frame_test
                compressed = compress_frame_test(
                    frame_data=channel.tobytes(),
                    width=w,
                    height=h,
                    padded_w=padded_w,
                    padded_h=padded_h,
                    block_size=block_size,
                    levels=128,
                    cosines=cosines
                )

                # Descomprime el canal usando decompress_frame_test
                decompressed = decompress_frame_test(
                    compressed,
                    w,
                    h,
                    block_size,
                    128,
                    cosines
                )

                # Verifica que la salida tenga la forma esperada
                self.assertEqual(decompressed.shape, (h, w), msg=f"Shape mismatch in channel {label} (frame {frame_idx})")
                self.assertTrue(np.issubdtype(decompressed.dtype, np.integer), msg=f"Dtype not integer in channel {label} (frame {frame_idx})")
                # El frame descomprimido no debe ser todo ceros
                self.assertTrue(np.any(decompressed), msg=f"All zeros in channel {label} (frame {frame_idx})")

                # Calcular diferencia absoluta media y máxima entre original y reconstruido
                abs_diff = np.abs(channel.astype(np.float32) - decompressed.astype(np.float32))
                mean_diff = np.mean(abs_diff)
                max_diff = np.max(abs_diff)
                print(f"[Frame {frame_idx} {label}] Mean abs diff: {mean_diff:.2f}, Max abs diff: {max_diff:.2f}")
                # El error medio debe ser razonable para compresión con pérdidas (ajustar umbral si necesario)
                self.assertLess(mean_diff, 30, msg=f"Mean diff too high in channel {label} (frame {frame_idx})")

                # Calcular PSNR entre bloque original y reconstruido
                mse = np.mean((channel.astype(np.float32) - decompressed.astype(np.float32)) ** 2)
                if mse == 0:
                    psnr = float('inf')
                else:
                    PIXEL_MAX = 255.0
                    psnr = 10 * np.log10((PIXEL_MAX ** 2) / mse)
                print(f"[Frame {frame_idx} {label}] PSNR: {psnr:.2f} dB")
                # El PSNR debe ser razonable para compresión con pérdidas (ajustar umbral si necesario)
                self.assertGreater(psnr, 20, msg=f"PSNR too low in channel {label} (frame {frame_idx})")
        

    def test_process_video_compression_five_frames(self):
        """
        Test process_video_compression y decompress_frame_test en los primeros 5 frames y canales.
        """
        comp = CompressorFinal()
        reader = BitWriter(self.video_path)

        # Lee el archivo completo como array 1D
        input_data = reader.read_original_video()
        frame_size = self.width * self.height * 3 // 2
        block_size = 8
        levels = 128
        cosines = precompute_cosines(block_size)

        # Comprime todo el video plano
        compressed_data = comp.process_video_compression(
            input_data,
            self.width,
            self.height,
            block_size,
            levels,
            cosines
        )
        first_5_compressed = compressed_data[:5]

        for i, frame_tuple in enumerate(first_5_compressed):
            # frame_tuple = [y_tuple, u_tuple, v_tuple]
            y_tuple, u_tuple, v_tuple = frame_tuple

            # Descomprime cada canal
            y_rec = decompress_frame_test(y_tuple, self.width, self.height, block_size, levels, cosines)
            u_rec = decompress_frame_test(u_tuple, self.width // 2, self.height // 2, block_size, levels, cosines)
            v_rec = decompress_frame_test(v_tuple, self.width // 2, self.height // 2, block_size, levels, cosines)

            # Frame original
            frame_start = i * frame_size
            frame_end = frame_start + frame_size
            orig_frame = input_data[frame_start:frame_end]
            yuv_orig = separate_yuv_compression(orig_frame, self.width, self.height)
            y_orig = yuv_orig[0]
            u_orig = yuv_orig[1]
            v_orig = yuv_orig[2]

            # Compara canal Y
            mse_y = np.mean((y_orig.astype(np.float32) - y_rec.astype(np.float32)) ** 2)
            psnr_y = float('inf') if mse_y == 0 else 10 * np.log10((255.0 ** 2) / mse_y)
            print(f"[Frame {i} Y] PSNR: {psnr_y:.2f} dB")

            # Compara canal U
            mse_u = np.mean((u_orig.astype(np.float32) - u_rec.astype(np.float32)) ** 2)
            psnr_u = float('inf') if mse_u == 0 else 10 * np.log10((255.0 ** 2) / mse_u)
            print(f"[Frame {i} U] PSNR: {psnr_u:.2f} dB")

            # Compara canal V
            mse_v = np.mean((v_orig.astype(np.float32) - v_rec.astype(np.float32)) ** 2)
            psnr_v = float('inf') if mse_v == 0 else 10 * np.log10((255.0 ** 2) / mse_v)
            print(f"[Frame {i} V] PSNR: {psnr_v:.2f} dB")

            # Puedes agregar asserts si quieres umbrales mínimos de calidad
            self.assertGreater(psnr_y, 20, f"PSNR Y muy bajo en frame {i}")
            self.assertGreater(psnr_u, 20, f"PSNR U muy bajo en frame {i}")
            self.assertGreater(psnr_v, 20, f"PSNR V muy bajo en frame {i}")
            
    def test_compress_decompress_block_full_on_first_frame(self):
        """
        Test compress_block_prediction_dct_and_quant y decompress_block_prediction_dct_and_quant
        en un bloque manual del canal Y del primer frame.
        """

        # Lee el archivo completo como array 1D
        input_data = np.fromfile(self.video_path, dtype=np.uint8)
        frame_size = self.width * self.height * 3 // 2
        first_frame = input_data[:frame_size]
        # Separa los canales YUV
        yuv_dict = separate_yuv_compression(first_frame, self.width, self.height)
        y_channel = yuv_dict[0]
        block_size = 8
        cosines = precompute_cosines(block_size)
        # Padding manual para el canal Y
        padded_h = ((self.height + block_size - 1) // block_size) * block_size
        padded_w = ((self.width + block_size - 1) // block_size) * block_size
        padded_y = np.zeros((padded_h, padded_w), dtype=np.uint8)
        padded_y[:self.height, :self.width] = y_channel
        # Extrae el primer bloque (0,0)
        by, bx = 0, 0
        y_start = by * block_size
        x_start = bx * block_size
        block = padded_y[y_start:y_start+block_size, x_start:x_start+block_size]
        # Llama compress_block_prediction_dct_and_quant
        quantized_dct_residual,reconstructed_block, mode_flag, min_val, step = compress_block_prediction_dct_and_quant(
            block,
            i_start=0,
            j_start=0,
            block_size=block_size,
            levels=128,
            cosines=cosines
        )
        # Verifica formas y tipos
        self.assertEqual(quantized_dct_residual.shape, (block_size, block_size))
        self.assertIsInstance(mode_flag, (int, np.integer))
        self.assertIsInstance(min_val, (float, np.floating))
        self.assertIsInstance(step, (float, np.floating))
        # El bloque cuantizado no debe ser todo ceros
        self.assertTrue(np.any(quantized_dct_residual))

        # Ahora descomprime el bloque y verifica que la forma y valores sean razonables
        decompressed_block = decompress_block_prediction_dct_and_quant(
            quantized_dct_residual,
            mode_flag,
            min_val,
            step,
            cosines,
            block_size
        )
        self.assertEqual(decompressed_block.shape, (block_size, block_size))
        # El bloque descomprimido debe ser similar al original (no necesariamente igual por pérdidas)
        # Pero al menos debe ser un array numérico válido
        self.assertTrue(np.issubdtype(decompressed_block.dtype, np.floating))

        # Calcular diferencia absoluta media y máxima entre original y reconstruido
        abs_diff = np.abs(block.astype(np.float32) - decompressed_block)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        print(f"Mean abs diff: {mean_diff:.2f}, Max abs diff: {max_diff:.2f}")
        # El error medio debe ser razonable para compresión con pérdidas (ajustar umbral si necesario)
        self.assertLess(mean_diff, 30)

        # Calcular PSNR entre bloque original y reconstruido
        mse = np.mean((block.astype(np.float32) - decompressed_block) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            PIXEL_MAX = 255.0
            psnr = 10 * np.log10((PIXEL_MAX ** 2) / mse)
        print(f"PSNR: {psnr:.2f} dB")
        # El PSNR debe ser razonable para compresión con pérdidas (ajustar umbral si necesario)
        self.assertGreater(psnr, 20)
        
    def test_compress_decompress_frame_on_first_frame(self):
        """
        Test compress_frame y decompress_frame en el canal Y del primer frame
        """
     
        # Lee el archivo completo como array 1D
        input_data = np.fromfile(self.video_path, dtype=np.uint8)
        frame_size = self.width * self.height * 3 // 2
        first_frame = input_data[:frame_size]
        # Separa los canales YUV
        yuv_dict = separate_yuv_compression(first_frame, self.width, self.height)
        y_channel = yuv_dict[0]
        block_size = 8
        cosines = precompute_cosines(block_size)
        # Padding manual para el canal Y
        padded_h = ((self.height + block_size - 1) // block_size) * block_size
        padded_w = ((self.width + block_size - 1) // block_size) * block_size
        # Comprime el canal Y
        compressed_y = compress_frame_test(
            frame_data=y_channel.tobytes(),
            width=self.width,
            height=self.height,
            padded_w=padded_w,
            padded_h=padded_h,
            block_size=block_size,
            levels=128,
            cosines=cosines
        )
        # Descomprime el canal Y
        decompressed_yuv = decompress_frame_test(
            compressed_y,
            self.width,
            self.height,
            block_size=block_size,
            levels=128,
            cosines=cosines
        )
        # Verifica que la salida tenga la forma esperada
     
        self.assertEqual(decompressed_yuv.shape, (self.height, self.width))
        self.assertTrue(np.issubdtype(decompressed_yuv.dtype, np.integer))
        # El frame descomprimido no debe ser todo ceros
        self.assertTrue(np.any(decompressed_yuv))
        
    def test_writer_for_first_frame(self):
        """
        Test BitWriter para escribir el primer frame comprimido en un archivo, luego leerlo, deserializarlo,
        descomprimirlo y comparar los frames originales y reconstruidos (PSNR).
        """
        # Lee el archivo completo como array 1D
        input_data = np.fromfile(self.video_path, dtype=np.uint8)
        frame_size = self.width * self.height * 3 // 2
        first_frame = input_data[:frame_size]
        block_size = 8
        levels = 128
        cosines = precompute_cosines(block_size)

        # Comprime el primer frame usando process_video_compression
        comp = CompressorFinal()
        compressed_frames = comp.process_video_compression(
            first_frame,
            self.width,
            self.height,
            block_size,
            levels,
            cosines
        )

        # Escribe el primer frame comprimido en un archivo temporal
        output_path = "test_videos/compressed/"
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "first_frame.finalcomp")
        writer = BitWriter(output_file)
        writer.write_compressed_video(compressed_frames[:1], self.width, self.height, 1, block_size, levels)

        # Lee los bytes escritos directamente del archivo
        with open(output_file, "rb") as f:
            file_bytes = f.read()

        # Serializa el frame comprimido en memoria usando la función
        serialized_bytes = comp.serialize_compressed_video(
            compressed_frames[:1], self.width, self.height, 1, block_size, levels
        )

        # Compara los bytes leídos del archivo con los bytes serializados en memoria
        self.assertEqual(file_bytes, serialized_bytes, "El archivo escrito no coincide con los datos serializados en memoria")

        # Deserializa y descomprime
        frames, width, height, num_frames, block_size, levels = comp.deserialize_compressed_video(file_bytes)
        decompressed_video = comp.process_video_decompression(frames, width, height, num_frames, block_size, levels, cosines)

        # Separa los canales originales y reconstruidos
        yuv_orig = separate_yuv_compression(first_frame, self.width, self.height)
        yuv_rec = separate_yuv_compression(decompressed_video, self.width, self.height)
        labels = ['Y', 'U', 'V']

        for i in range(3):
            orig = yuv_orig[i].astype(np.float32)
            rec = yuv_rec[i].astype(np.float32)
            mse = np.mean((orig - rec) ** 2)
            psnr = float('inf') if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
            print(f"[First frame {labels[i]}] PSNR: {psnr:.2f} dB")
            self.assertGreater(psnr, 20, f"PSNR too low in channel {labels[i]} of first frame")

    def yuv420_to_rgb_and_save(self, yuv_frame, width, height, output_path):
        """
        Convierte un frame YUV420 plano (Y luego U luego V, todos uint8) a RGB y lo guarda como imagen PNG.
        - yuv_frame: array 1D o bytes, tamaño = width*height*3//2
        - width, height: dimensiones del frame
        - output_path: ruta donde guardar la imagen (ej: 'frame.png')
        """
        yuv = np.frombuffer(yuv_frame, dtype=np.uint8)
        y_size = width * height
        uv_size = (width // 2) * (height // 2)

        Y = yuv[0:y_size].reshape((height, width))
        U = yuv[y_size:y_size + uv_size].reshape((height // 2, width // 2))
        V = yuv[y_size + uv_size:].reshape((height // 2, width // 2))

        # Upsample U y V a tamaño de Y
        U_up = U.repeat(2, axis=0).repeat(2, axis=1)
        V_up = V.repeat(2, axis=0).repeat(2, axis=1)

        # Convertir a float para el cálculo
        Y = Y.astype(np.float32)
        U = U_up.astype(np.float32) - 128.0
        V = V_up.astype(np.float32) - 128.0

        # Conversión YUV a RGB (BT.601)
        R = Y + 1.402 * V
        G = Y - 0.344136 * U - 0.714136 * V
        B = Y + 1.772 * U

        # Clip y convertir a uint8
        RGB = np.stack([R, G, B], axis=-1)
        RGB = np.clip(RGB, 0, 255).astype(np.uint8)

        # Guardar como imagen
        img = Image.fromarray(RGB, 'RGB')
        img.save(output_path)

    def serialize_compressed_video(self, compressed_frames, block_size):
        """
        Serializa la estructura de salida de process_video_compression a un array 1D np.uint8.
        """
        byte_chunks = []
        for frame in compressed_frames:
            for channel_tuple in frame:  # [y_out, u_out, v_out]
                processed_blocks, mode_flags, min_vals, steps = channel_tuple
                n_blocks_y, n_blocks_x = mode_flags.shape
                for by in range(n_blocks_y):
                    for bx in range(n_blocks_x):
                        arr = processed_blocks[by, bx].astype(np.uint8)
                        mode_flag = int(mode_flags[by, bx])
                        min_val = float(min_vals[by, bx])
                        step = float(steps[by, bx])
                        # Serializa metadatos y bloque
                        byte_chunks.append(struct.pack("Bff", mode_flag, min_val, step))
                        byte_chunks.append(arr.tobytes())
        # Une todo y convierte a np.ndarray
        all_bytes = b"".join(byte_chunks)
        return np.frombuffer(all_bytes, dtype=np.uint8)
