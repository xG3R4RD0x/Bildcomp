from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import sys

if len(sys.argv) < 2:
    exit(f"Usage: {sys.argv[0]} <video_WxH.yuv>")

FILENAME_YUV = sys.argv[1]
WIDTH, HEIGHT = FILENAME_YUV.split('.', 1)[0].split("_")[-1].split('x')
WIDTH = int(WIDTH)
HEIGHT = int(HEIGHT)
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2  # YUV420p
FPS = 24

frames = []
frame_index = 0
with open(FILENAME_YUV, 'rb') as f:
    def read_yuv_frame():
        data = f.read(FRAME_SIZE)
        if len(data) < FRAME_SIZE:
            return None

        y = np.frombuffer(data[0:WIDTH*HEIGHT], dtype=np.uint8).reshape((HEIGHT, WIDTH))
        u = np.frombuffer(data[WIDTH*HEIGHT:WIDTH*HEIGHT + (WIDTH//2)*(HEIGHT//2)], dtype=np.uint8).reshape((HEIGHT//2, WIDTH//2))
        v = np.frombuffer(data[WIDTH*HEIGHT + (WIDTH//2)*(HEIGHT//2):], dtype=np.uint8).reshape((HEIGHT//2, WIDTH//2))

        # Upsampling U und V
        u_up = u.repeat(2, axis=0).repeat(2, axis=1)
        v_up = v.repeat(2, axis=0).repeat(2, axis=1)

        # YUV → RGB
        yuv = np.stack((y, u_up, v_up), axis=2).astype(np.float32)
        yuv[:, :, 0] = yuv[:, :, 0] - 16
        yuv[:, :, 1] = yuv[:, :, 1] - 128
        yuv[:, :, 2] = yuv[:, :, 2] - 128

        m = np.array([[1.0,  0.0,       1.402],
                    [1.0, -0.34414, -0.71414],
                    [1.0,  1.772,    0.0]])
        rgb = yuv @ m.T
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, 'RGB')

    while frame := read_yuv_frame():
        frames.append(frame)

# GUI Setup
root = tk.Tk()
root.title(f"BildKomp - {FILENAME_YUV}")
label = tk.Label(root)
label.pack()

def update_frame():
    global frame_index

    root.title(f"[{frame_index}/{len(frames)}] BildKomp - {FILENAME_YUV}")

    img = ImageTk.PhotoImage(frames[frame_index])
    label.config(image=img)
    label.image = img
    frame_index = (frame_index + 1) % len(frames)
    root.after(1000 // FPS, update_frame)

update_frame()
root.mainloop()