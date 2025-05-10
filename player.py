import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import sys
import os
import time
from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage

FPS = 24
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Player:
    def __init__(self, root, filename=None):
        self.root = root
        self.filename = filename
        self.frames = []
        self.frame_index = 0
        self.width = 0
        self.height = 0
        self.video_id = 0
        self.stop_update_loop = False
        self.video_is_playing = False

        self.root.geometry("600x600")
        self.root.title("BildKomp")

        self.build_gui()

        if filename:
            self.load_yuv_file(filename)

    def build_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side="top", fill="x")

        file_menu_btn = tk.Menubutton(top_frame, text="File", relief=tk.RAISED)
        file_menu = tk.Menu(file_menu_btn, tearoff=0)
        file_menu.add_command(label="Load", command=self.menu_load_file)
        file_menu.add_command(label="Save as ...", command=self.menu_save_file)
        file_menu_btn.config(menu=file_menu)
        file_menu_btn.pack(side="left", padx=0, pady=5)

        option_menu_btn = tk.Menubutton(top_frame, text="Option", relief=tk.RAISED)
        option_menu = tk.Menu(option_menu_btn, tearoff=0)
        option_menu.add_command(label="Quantisation", command=lambda: print("Option 1"))
        option_menu.add_command(label="Placeholder", command=lambda: print("Option 2"))
        option_menu_btn.config(menu=option_menu)
        option_menu_btn.pack(side="left", padx=0, pady=5)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        self.image_player = tk.Label(left_frame)
        self.image_player.pack()

        right_frame = tk.Frame(main_frame, width=200)
        right_frame.pack(side="right", fill="y")
        tk.Label(
            right_frame, text="Info-Bereich\n(in Entwicklung)", anchor="center"
        ).pack(pady=20)

    def menu_load_file(self):
        path = filedialog.askopenfilename(
            initialdir=ROOT_DIR, filetypes=[("YUV files", "*.yuv")]
        )
        if path:
            self.load_yuv_file(path)

    def load_yuv_file(self, path):
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            return

        self.video_id += 1
        # self.stop_update_loop = True
        # current_id = self.video_id

        # Placeholder anzeigen
        self.image_player.config(
            image="", text="Loading Video...", font=("Arial", 20), compound="center"
        )
        self.image_player.image = None
        self.root.update()

        try:
            dims = path.split(".", 1)[0].split("_")[-1].split("x")
            self.width, self.height = int(dims[0]), int(dims[1])
        except Exception as e:
            print(f"Error parsing resolution from filename: {e}")
            return

        self.frames = self.load_yuv_frames(path)
        self.frame_index = 0
        self.filename = path
        self.root.title(f"BildKomp - {os.path.basename(path)}")

        self.update_frame()

    def load_yuv_frames(self, path):
        frames = []
        frame_size = self.width * self.height * 3 // 2  # YUV420p

        with open(path, "rb") as f:
            while True:
                data = f.read(frame_size)
                if len(data) < frame_size:
                    break

                # Call separate_yuv from DecorrelationStage
                components = DecorrelationStage.separate_yuv(
                    data, self.width, self.height
                )
                rgb = components["rgb"]
                frames.append(Image.fromarray(rgb, "RGB"))

        return frames

    def update_frame(self):
        if not self.frames:
            return

        self.root.title(
            f"[{self.frame_index}/{len(self.frames)}] BildKomp - {os.path.basename(self.filename)}"
        )

        img = ImageTk.PhotoImage(self.frames[self.frame_index])
        self.image_player.config(image=img, text="", compound=None)
        self.image_player.image = img
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.root.after(1000 // FPS, self.update_frame)

    def menu_save_file(self):
        pass


if __name__ == "__main__":
    filename_arg = sys.argv[1] if len(sys.argv) >= 2 else None
    root = tk.Tk()
    app = Player(root, filename_arg)
    root.mainloop()
