import numpy as np
import os
import sys
import mvp
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from PIL import Image, ImageTk
from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage
from pipeline.compression.compressor_final import CompressorFinal

FPS = 24
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Video:
    def __init__(self, id, path):
        self.id = id
        self.path = path
        self.width = 0
        self.height = 0
        self.frames = []
        self.bitrate_per_frame = []
        self._load()

    def calculate_bitrate(self):
        FrameSize_bytes = self.width * self.height * 1.5 # 1.5 because 1 Y + 1/4 U + 1/4 V
        bitrate = FrameSize_bytes * 8
        self.bitrate_per_frame.append(bitrate)

    def _load(self):
        '''
        Distinction between .yuv and .vid files
        '''
        ds = DecorrelationStage()
        if self.path.endswith(".yuv"): 
            ''' YUV File handling '''
            dims = self.path.split(".", 1)[0].split("_")[-1].split("x")
            self.width, self.height = int(dims[0]), int(dims[1])
            
            t_start = time.time()
            frame_size = self.width * self.height * 3 // 2  # YUV420p

            with open(self.path, "rb") as f:
                while True:
                    data = f.read(frame_size)
                    if len(data) < frame_size:
                        break
                    components = ds.separate_yuv(data, self.width, self.height)
                    self.calculate_bitrate()
                    rgb = components["rgb"]
                    img = Image.fromarray(rgb, "RGB")
                    self.frames.append(img)
            t_end = time.time()
            print(f"Decoding time: {t_end - t_start:.8f} seconds")
        elif self.path.endswith(".vid"): 
            ''' VID file handling '''
            dims = self.path.split(".", 1)[0].split("_")[-1].split("x")
            self.width, self.height = int(dims[0]), int(dims[1])

            t_start = time.time()
            frame_size = self.width * self.height * 3 // 2  # YUV420p
            frames, bits_per_frame = mvp.decompress(self.path)
            self.bitrate_per_frame = bits_per_frame

            for frame in frames:
                yuv_bytestring = frame.y + frame.u + frame.v
                components = ds.separate_yuv(yuv_bytestring, self.width, self.height)
                rgb = components["rgb"]
                img = Image.fromarray(rgb, "RGB")
                self.frames.append(img)
            t_end = time.time()
            print(f"Decoding time: {t_end - t_start:.8f} seconds")
        elif self.path.endswith(".finalcomp"):
            ''' Handling for Predcition algorithmus '''
            dims = self.path.split(".", 1)[0].split("_")[-1].split("x")
            self.width, self.height = int(dims[0]), int(dims[1])

            t_start = time.time()
            frame_size = self.width * self.height * 3 // 2  # YUV420p
            frames, bits_per_frame = mvp.load_vid_file_in_player(self.path)
            self.bitrate_per_frame = bits_per_frame

            for frame in frames:
                yuv_bytestring = frame.y + frame.u + frame.v
                components = ds.separate_yuv(yuv_bytestring, self.width, self.height)
                rgb = components["rgb"]
                img = Image.fromarray(rgb, "RGB")
                self.frames.append(img)
            t_end = time.time()
            print(f"Decoding time: {t_end - t_start:.8f} seconds")

class RadioDialog(simpledialog.Dialog):
    def __init__(self, parent, title, options):
        self.options = options
        self.selected = tk.StringVar(value=options[0])
        super().__init__(parent, title)

    def body(self, master):
        tk.Label(master, text="Choose an option:").pack()
        for opt in self.options:
            tk.Radiobutton(master, text=opt, variable=self.selected, value=opt).pack(anchor="w")

    def apply(self):
        self.result = self.selected.get()

class Player:
    def __init__(self, root, filename=None):
        self.root = root
        self.video_left = None
        self.video_right = None
        self.frame_index_left = 0
        self.frame_index_right = 0
        self.video_id = 0
        self.is_playing = False
        self.after_id = None
        self.quantization_level = 10

        self.psnr_values = []

        self.root.geometry("500x400")
        self.root.title("BildKomp")
        self.build_gui()

        if filename:
            self.load_video_file(filename, side="left")

    def build_gui(self):
        '''
        Builds all the UI elements in the GUI and arranges them
        '''
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side="top", anchor="nw", fill="x")

        file_menu_btn = tk.Menubutton(self.top_frame, text="File", relief=tk.RAISED)
        file_menu = tk.Menu(file_menu_btn, tearoff=0)
        file_menu.add_command(label="Load Left Video", command=lambda: self.menu_load_file("left"))
        file_menu.add_command(label="Load Right Video", command=lambda: self.menu_load_file("right"))
        file_menu.add_command(label="Save left side as...", command=lambda: self.menu_save_file("left"))
        file_menu.add_command(label="Save right side as...", command=lambda: self.menu_save_file("right"))
        file_menu_btn.config(menu=file_menu)
        file_menu_btn.pack(side="left", padx=2, pady=2)

        options_menu_btn = tk.Menubutton(self.top_frame, text="Options", relief=tk.RAISED)
        options_menu = tk.Menu(options_menu_btn, tearoff=0)
        options_menu.add_command(label="Chosse encoding algorithm", command=self.set_quantization_level)
        options_menu_btn.config(menu=options_menu)
        options_menu_btn.pack(side="left", padx=0, pady=0)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side="top", anchor="nw", fill="x", expand=True)

        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side="top", anchor= "nw", padx=0)
        self.left_video_frame = tk.Frame(self.video_frame)
        self.left_video_frame.pack(side="left", anchor="nw", padx=0, pady=0)
        self.right_video_frame = tk.Frame(self.video_frame)
        self.right_video_frame.pack(side="left", anchor="nw", padx=0, pady=0)

        self.image_player_left = tk.Label(self.left_video_frame)
        self.image_player_left.pack(anchor="nw")
        self.placeholder_left = tk.Label(
            self.left_video_frame,
            width=20,
            height=10,
            text="Left Side \n(The original YUV file)",
            font=("Arial", 14),
            bg="#E0E0E0",
            relief="ridge"
        )
        self.placeholder_left.pack()

        self.image_player_right = tk.Label(self.right_video_frame)
        self.image_player_right.pack(anchor="nw")
        self.placeholder_right = tk.Label(
            self.right_video_frame,
            width=20,
            height=10,
            text="Right Side \n(Our commpressed VID file)",
            font=("Arial", 14),
            bg="#E0E0E0",
            relief="ridge"
        )
        self.placeholder_right.pack()


        self.control_frame = tk.Frame(self.left_video_frame)
        self.start_btn = tk.Button(self.control_frame, text="|<", command=self.jump_to_start)
        self.prev_btn = tk.Button(self.control_frame, text= "<<", command=self.step_back)
        self.play_btn = tk.Button(self.control_frame, text="Play", width=5, command=self.toggle_play)
        self.next_btn = tk.Button(self.control_frame, text= ">>", command=self.step_forward)
        self.end_btn = tk.Button(self.control_frame, text=">|", command=self.jump_to_end)

        
        self.info_frame = tk.Frame(self.main_frame, width=0)
        self.left_info_frame = tk.Frame(self.info_frame, width=500)
        self.psnr_label = tk.Label(self.left_info_frame, text="PSNR: -")
        self.bitrate_label_left = tk.Label(self.left_info_frame, text="Bitrate left side: -")
        self.bitrate_label_right = tk.Label(self.left_info_frame, text="Bitrate right side: -")
        self.frame_label_left = tk.Label(self.left_info_frame, text="Frame left side: -")
        self.frame_label_right = tk.Label(self.left_info_frame, text="Frame right side: -")
        #self.right_info_frame = tk.Frame(self.info_frame, width=300, background="red")
        #self.right_info_frame.pack(side="left", anchor="nw", padx=0)

    def show_controls(self):
        '''
        Show some GUI elements (play, pause,...) only when a Video is loaded in the player
        '''
        self.control_frame.pack(pady=2, anchor="nw")
        self.start_btn.pack(side="left", padx=2)
        self.prev_btn.pack(side="left", padx=2)
        self.play_btn.pack(side="left", padx=2)
        self.next_btn.pack(side="left", padx=2)
        self.end_btn.pack(side="left", padx=2)

    def show_info_panel(self):
        self.info_frame.pack(side="top", anchor="nw", padx=0)
        self.left_info_frame.pack(side="left", anchor="nw")
        self.psnr_label.pack(anchor='nw')
        self.bitrate_label_left.pack(anchor='nw')
        self.bitrate_label_right.pack(anchor='nw')
        self.frame_label_left.pack(anchor="nw", padx=0)
        self.frame_label_right.pack(anchor="nw", padx=0)

    def hide_controls(self):
        self.control_frame.pack_forget()

    def menu_load_file(self, side):
        path = filedialog.askopenfilename(
            initialdir=ROOT_DIR,
            filetypes=[("All files", "*.*")]
        )
        if path:
            self.load_video_file(path, side=side)
            # if format == "yuv":
            #     self.load_yuv_file(path)
            # elif format == "vid":
            #     self.load_vid_file(path)

    def set_quantization_level(self):
        value = simpledialog.askfloat("Quantization Level", "Enter quantization level:", minvalue=1.0, maxvalue=100.0)
        if value is not None:
            self.quantization_level = value
        print(self.quantization_level)
        return value

    def load_video_file(self, path, side):
        '''
        Handle if the video is shown on the left or right side
        '''
        if side == "left":
            self.placeholder_left.destroy()
            self.image_player_left.config(image="", text="Loading...", font=("Arial", 20), compound="center")
            self.root.update()
            try:
                self.video_left = Video(self.video_id, path)
                self.frame_index_left = 0
            except Exception as e:
                print(f"Left video failed: {e}")
                return
        elif side == "right":
            self.placeholder_right.destroy()
            self.image_player_right.config(image="", text="Loading...", font=("Arial", 20), compound="center")
            self.root.update()
            try:
                self.video_right = Video(self.video_id + 1, path)
                self.frame_index_right = 0
            except Exception as e:
                print(f"Right video failed: {e}")
                return
            
        if self.video_left and self.video_right:
            self.precalculate_psnr()
            
        self.video_id += 2
        self.jump_to_start()
        self.display_frames()
        self.show_controls()
        self.show_info_panel()

    def display_frames(self):
        '''
        Updates the UI elements to show the videoframe and its current statistics
        '''
        if self.video_left and self.video_left.frames:
            frame = self.video_left.frames[self.frame_index_left]
            img = ImageTk.PhotoImage(frame)
            self.image_player_left.config(image=img, text="", compound=None)
            self.image_player_left.image = img
            
            # Show video statistics
            frame_info_left = f"Frame left side: {self.frame_index_left if self.video_left else '-'} / "
            frame_info_left += f"{(len(self.video_left.frames) - 1) if self.video_left else '-'}"
            self.frame_label_left.config(text=frame_info_left)

            # Bitrate
            bitrate = self.video_left.bitrate_per_frame[self.frame_index_left] / 1000
            self.bitrate_label_left.config(text=f"Bitrate left side: {bitrate:.2f} kbpf")

        if self.video_right and self.video_right.frames:
            frame = self.video_right.frames[self.frame_index_right]
            img = ImageTk.PhotoImage(frame)
            self.image_player_right.config(image=img, text="", compound=None)
            self.image_player_right.image = img

            # Show video statistics
            frame_info_right = f"Frame right side: {self.frame_index_right if self.video_right else '-'} / "
            frame_info_right += f"{(len(self.video_right.frames) - 1) if self.video_right else '-'}"
            self.frame_label_right.config(text=frame_info_right)

            # Bitrate
            bitrate = self.video_right.bitrate_per_frame[self.frame_index_right] / 1000
            self.bitrate_label_right.config(text=f"Bitrate right side: {bitrate:.2f} kbpf")

        if self.psnr_values:
            if len(self.video_right.frames) > len(self.video_left.frames):
                psnr = self.psnr_values[self.frame_index_right]
            else:
                psnr = self.psnr_values[self.frame_index_left]
            self.psnr_label.config(text=f"PSNR: {psnr:.2f} dB")

    def playback_loop(self):
        '''
        Recursive function call for playing the next frame in the video.
        Is toggled on and off with the play/pause button.
        '''
        if not self.is_playing:
            return

        if self.video_left:
            self.frame_index_left = (self.frame_index_left + 1) % len(self.video_left.frames)
        if self.video_right:
            self.frame_index_right = (self.frame_index_right + 1) % len(self.video_right.frames)

        self.display_frames()
        self.after_id = self.root.after(1000 // FPS, self.playback_loop)

    def start_playback(self):
        if not (self.video_left or self.video_right):
            return
        self.play_btn.config(text="Pause")
        self.is_playing = True
        self.playback_loop()

    def stop_playback(self):
        self.play_btn.config(text="Play")
        self.is_playing = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def step_back(self):
        self.stop_playback()
        if self.video_left:
            self.frame_index_left = (self.frame_index_left - 1) % len(self.video_left.frames)
        if self.video_right:
            self.frame_index_right = (self.frame_index_right - 1) % len(self.video_right.frames)
        self.display_frames()

    def step_forward(self):
        self.stop_playback()
        if self.video_left:
            self.frame_index_left = (self.frame_index_left + 1) % len(self.video_left.frames)
        if self.video_right:
            self.frame_index_right = (self.frame_index_right + 1) % len(self.video_right.frames)
        self.display_frames()

    def jump_to_start(self):
        self.stop_playback()
        if self.video_left:
            self.frame_index_left = 0
        if self.video_right:
            self.frame_index_right = 0
        self.display_frames()

    def jump_to_end(self):
        self.stop_playback()
        if self.video_left:
            self.frame_index_left = len(self.video_left.frames) -1
        if self.video_right:
            self.frame_index_right = len(self.video_right.frames) -1
        self.display_frames()

    def toggle_play(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def menu_save_file(self, side):
        '''
        Opens compression settings and then save a file into a specified folder
        '''
        if side == "left":
            video_to_save = self.video_left
        elif side == "right":
            video_to_save = self.video_right

        if video_to_save:
            list_of_comps = ["DCT", "Prediction"]
            comp_dialog = RadioDialog(root, "Select a compression algorithm", list_of_comps)
            print("Selected:", comp_dialog.result)
            if comp_dialog.result == list_of_comps[0]: # dct
                value = self.set_quantization_level()
                if value is not None:
                    self.quantization_level = value
                    output_filepath = filedialog.asksaveasfilename(
                        defaultextension=".vid",
                        filetypes=[("Vid files", "*.vid"), ("All files", "*.*" )],
                        title="Save file as"
                    )
                    if output_filepath:
                        t_start = time.time()
                        print("Compressing...")
                        mvp.compress_and_save_to_file(video_to_save.path, output_filepath, self.quantization_level)
                        t_end = time.time()
                        print(f"Encoding time: {t_end - t_start:.8f} seconds")
            elif comp_dialog.result == list_of_comps[1]: # prediction
                value = self.set_quantization_level()
                if value is not None:
                    self.quantization_level = int(value)
                    output_filepath = filedialog.askdirectory()
                    output_filepath = output_filepath + "/"

                    if output_filepath:
                        t_start = time.time()
                        print("Compressing...")
                        cf = CompressorFinal()
                        cf.compress_video(video_path=video_to_save.path, output_path=output_filepath, height= video_to_save.height, width=video_to_save.width, block_size=8, levels=self.quantization_level)
                        #mvp.compress_and_save_to_file(video_to_save.path, output_filepath, self.quantization_level)
                        t_end = time.time()
                        print(f"Encoding time: {t_end - t_start:.8f} seconds")
        else: 
            print(f"Player on {side} side does not contain a file")

    def precalculate_psnr(self) -> list:
        '''
        PSNR is always pre calculated when two videos are in the player. 
        The 
        '''
        self.psnr_values = []
        min_len = min(len(self.video_left.frames), len(self.video_right.frames))
        max_len = max(len(self.video_left.frames), len(self.video_right.frames))
        for i in range(min_len):
            psnr = self.calculate_psnr_of_frame(self.video_left.frames[i], self.video_right.frames[i])
            self.psnr_values.append(psnr)

        for i in range(min_len, max_len):
            self.psnr_values.append(0.0)

    def calculate_psnr_of_frame(self, frame_left, frame_right):
        arr1 = np.asarray(frame_left).astype(np.float32)
        arr2 = np.asarray(frame_right).astype(np.float32)
        mse = np.mean((arr1 - arr2) ** 2)
        return 20 * np.log10(255.0) - 10 * np.log10(mse)

if __name__ == "__main__":
    filename_arg = sys.argv[1] if len(sys.argv) >= 2 else None
    root = tk.Tk()
    app = Player(root, filename_arg)
    root.mainloop()

