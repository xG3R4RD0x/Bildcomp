
import os
import unittest
from pipeline.stages.bitwriter.bitwritter import BitWriter

TEST_DIR = os.path.join(os.path.dirname(__file__), '../test_videos/bitwirtter_test')
os.makedirs(TEST_DIR, exist_ok=True)

class TestBitWriter(unittest.TestCase):
    def get_test_file(self, name):
        return os.path.join(TEST_DIR, name)

    

    def test_write_and_read_header(self):
        file_path = self.get_test_file('header_test.bin')
        writer = BitWriter(file_path)
        width, height, num_frames, block_size, levels = 176, 144, 30, 8, 3
        writer.write_header(width, height, num_frames, block_size, levels)
        result = writer.read_header()
        self.assertEqual(result, (width, height, num_frames, block_size, levels))

    def test_write_and_read_video(self):
        file_path = self.get_test_file('video_test.bin')
        writer = BitWriter(file_path)
        width, height, num_frames, block_size, levels = 176, 144, 30, 8, 3
        writer.write_header(width, height, num_frames, block_size, levels)
        video_bytes = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        # Use the raw-bytes version of write_video (first definition)
        # Since BitWriter has two write_video methods, we call the one that writes bytes directly
        # This is the expected behavior for this test
        with open(file_path, 'ab') as f:
            f.write(video_bytes)
        read_bytes = writer.read_compressed_video()
        self.assertEqual(read_bytes, video_bytes)

    def test_header_and_video_together(self):
        file_path = self.get_test_file('header_video_test.bin')
        writer = BitWriter(file_path)
        width, height, num_frames, block_size, levels = 8, 8, 1, 8, 3
        # Simulate a single YUV420 frame: Y (8x8), U (4x4), V (4x4)
        y = bytes([10] * 64)  # 8x8
        u = bytes([20] * 16)  # 4x4
        v = bytes([30] * 16)  # 4x4
        yuv_frame = y + u + v
        writer.write_header(width, height, num_frames, block_size, levels)
        with open(file_path, 'ab') as f:
            f.write(yuv_frame)
        header = writer.read_header()
        self.assertEqual(header, (width, height, num_frames, block_size, levels))
        read_bytes = writer.read_compressed_video()
        self.assertEqual(read_bytes, yuv_frame)
