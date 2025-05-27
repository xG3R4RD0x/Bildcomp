import unittest
import os

from pipeline.compression.compressor import Compressor


class TestCompressor(unittest.TestCase):
    def setUp(self):
        self.video_path = "test_videos/Sign_Irene_352x288.yuv"
        self.video_path_compressed = "test_videos/compressed/Sign_Irene_352x288.decp"
        self.video_path_compressed_dct = "test_videos/compressed/Sign_Irene_352x288.dect"
        self.video_path_decompressed = "test_videos/decompressed/Sign_Irene_352x288.yuv"
        self.width = 352
        self.height = 288

    def test_compressor(self):
        compressor = Compressor()
        output_path = compressor.compress_video(
            self.height, self.width, self.video_path
        )

        # Check if the output file exists
        self.assertTrue(os.path.isfile(output_path), "Compressed file not found.")

        # Check if the output file is not empty
        self.assertGreater(os.path.getsize(output_path), 0, "Compressed file is empty.")
 

    def test_decompressor(self):
        """Test the compression and decompression process to ensure file integrity."""
        compressor = Compressor()

        # First compress the video
        compressed_path = self.video_path_compressed
        self.assertTrue(os.path.isfile(compressed_path), "Compressed file not created")

        # Decompress the file
        decompressed_path = compressor.decompress_video(compressed_path)

        # Verify the decompressed file exists
        self.assertTrue(
            os.path.isfile(decompressed_path),
            f"Decompressed file not found at {decompressed_path}",
        )

        # Verify the decompressed file is not empty
        self.assertGreater(
            os.path.getsize(decompressed_path), 0, "Decompressed file is empty"
        )

        # Compare file sizes (should be the same for YUV420 format)
        original_size = os.path.getsize(self.video_path)
        decompressed_size = os.path.getsize(decompressed_path)

        # Log file size details to help diagnose the issue
        print(f"Original file size: {original_size} bytes")
        print(f"Decompressed file size: {decompressed_size} bytes")
        print(f"Difference: {original_size - decompressed_size} bytes")

        self.assertEqual(
            original_size,
            decompressed_size,
            f"File size mismatch: original={original_size} bytes, decompressed={decompressed_size} bytes",
        )

        # Validate content by checking a sample of bytes
        with open(self.video_path, "rb") as f_orig, open(
            decompressed_path, "rb"
        ) as f_decomp:
            # Read the entire files
            orig_data = f_orig.read()
            decomp_data = f_decomp.read()

            # Check file lengths
            print(f"Original read length: {len(orig_data)} bytes")
            print(f"Decompressed read length: {len(decomp_data)} bytes")

            # Find the position where differences start (if any)
            min_len = min(len(orig_data), len(decomp_data))
            for i in range(min_len):
                if orig_data[i] != decomp_data[i]:
                    print(f"First difference at byte position {i}")
                    print(
                        f"Original byte: {orig_data[i]}, Decompressed byte: {decomp_data[i]}"
                    )
                    break

            # Read the first 1000 bytes (or whole file if smaller)
            sample_size = min(1000, original_size)
            f_orig.seek(0)
            f_decomp.seek(0)
            orig_data = f_orig.read(sample_size)
            decomp_data = f_decomp.read(sample_size)

            # Check if the samples match
            self.assertEqual(
                orig_data,
                decomp_data,
                "Content of decompressed file doesn't match the original",
            )

            # Optional: check random positions in larger files
            if original_size > 10000:
                # Check middle of file
                mid_pos = original_size // 2
                f_orig.seek(mid_pos)
                f_decomp.seek(mid_pos)
                self.assertEqual(
                    f_orig.read(100),
                    f_decomp.read(100),
                    "Content mismatch in middle of file",
                )

                # Check end of file
                end_pos = max(0, original_size - 1000)
                f_orig.seek(end_pos)
                f_decomp.seek(end_pos)
                self.assertEqual(
                    f_orig.read(), f_decomp.read(), "Content mismatch at end of file"
                )

        print(f"Compression-decompression validation successful for {self.video_path}")
    def test_compressor_decompressor_dct(self):
        """Test the compression and decompression process to ensure file integrity."""
        compressor = Compressor()

        # First compress the video
        
        compressed_path = compressor.compress_video_dct(
            self.height, self.width, self.video_path
        )
        
        
        # Decompress the file
        decompressed_path = compressor.decompress_video_with_dct(compressed_path)

        # Verify the decompressed file exists
        self.assertTrue(
            os.path.isfile(decompressed_path),
            f"Decompressed file not found at {decompressed_path}",
        )

        # Verify the decompressed file is not empty
        self.assertGreater(
            os.path.getsize(decompressed_path), 0, "Decompressed file is empty"
        )

        # Compare file sizes (should be the same for YUV420 format)
        original_size = os.path.getsize(self.video_path)
        decompressed_size = os.path.getsize(decompressed_path)

        # Log file size details to help diagnose the issue
        print(f"Original file size: {original_size} bytes")
        print(f"Decompressed file size: {decompressed_size} bytes")
        print(f"Difference: {original_size - decompressed_size} bytes")

        self.assertEqual(
            original_size,
            decompressed_size,
            f"File size mismatch: original={original_size} bytes, decompressed={decompressed_size} bytes",
        )

        # Validate content by checking a sample of bytes
        with open(self.video_path, "rb") as f_orig, open(
            decompressed_path, "rb"
        ) as f_decomp:
            # Read the entire files
            orig_data = f_orig.read()
            decomp_data = f_decomp.read()

            # Check file lengths
            print(f"Original read length: {len(orig_data)} bytes")
            print(f"Decompressed read length: {len(decomp_data)} bytes")

            # Find the position where differences start (if any)
            min_len = min(len(orig_data), len(decomp_data))
            for i in range(min_len):
                if orig_data[i] != decomp_data[i]:
                    print(f"First difference at byte position {i}")
                    print(
                        f"Original byte: {orig_data[i]}, Decompressed byte: {decomp_data[i]}"
                    )
                    break

            # Read the first 1000 bytes (or whole file if smaller)
            sample_size = min(1000, original_size)
            f_orig.seek(0)
            f_decomp.seek(0)
            orig_data = f_orig.read(sample_size)
            decomp_data = f_decomp.read(sample_size)

            # Check if the samples match
            self.assertEqual(
                orig_data,
                decomp_data,
                "Content of decompressed file doesn't match the original",
            )

            # Optional: check random positions in larger files
            if original_size > 10000:
                # Check middle of file
                mid_pos = original_size // 2
                f_orig.seek(mid_pos)
                f_decomp.seek(mid_pos)
                self.assertEqual(
                    f_orig.read(100),
                    f_decomp.read(100),
                    "Content mismatch in middle of file",
                )

                # Check end of file
                end_pos = max(0, original_size - 1000)
                f_orig.seek(end_pos)
                f_decomp.seek(end_pos)
                self.assertEqual(
                    f_orig.read(), f_decomp.read(), "Content mismatch at end of file"
                )

        print(f"Compression-decompression validation successful for {self.video_path}")

    def test_compare_original_and_decompressed_frame_by_frame(self):
        """Test to compare the decompressed file with the original file frame by frame."""

        frame_size = self.width * self.height * 3 // 2  # YUV420 frame size

        with open(self.video_path, "rb") as original_file, open(
            self.video_path_decompressed, "rb"
        ) as decompressed_file:
            frame_index = 0
            original_frames = 0
            decompressed_frames = 0

            while True:
                original_frame = original_file.read(frame_size)
                decompressed_frame = decompressed_file.read(frame_size)

                # Count frames from each file
                if original_frame:
                    original_frames += 1
                if decompressed_frame:
                    decompressed_frames += 1

                # If one file has data but the other doesn't, we have a frame count mismatch
                if bool(original_frame) != bool(decompressed_frame):
                    print(f"Frame count mismatch detected at frame index {frame_index}")
                    print(f"Original file has data: {bool(original_frame)}")
                    print(f"Decompressed file has data: {bool(decompressed_frame)}")
                    break

                # If both files are empty, we've reached the end successfully
                if not original_frame and not decompressed_frame:
                    break

                # Assert that the frames match
                self.assertEqual(
                    original_frame,
                    decompressed_frame,
                    f"Frame {frame_index} does not match between the original and decompressed files.",
                )
                print(
                    f"Frame {frame_index} matches between the original and decompressed files."
                )
                frame_index += 1

            # After loop, verify frame counts match
            print(f"Total frames in original file: {original_frames}")
            print(f"Total frames in decompressed file: {decompressed_frames}")

            self.assertEqual(
                original_frames,
                decompressed_frames,
                f"Frame count mismatch: original={original_frames}, decompressed={decompressed_frames}",
            )
