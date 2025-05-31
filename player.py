import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import sys
from pipeline.stages.decorrelation.decorrelation_stage import DecorrelationStage

FPS = 24
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Video:
    def __init__(self, id, path):
        self.id = id
        self.path = path
        self.width = 0
        self.height = 0
        self.frames = []
        self._load()

    def _load(self):
        '''
        Here the distinction between .yuv and .vid files need to be made 
        '''
        dims = self.path.split(".", 1)[0].split("_")[-1].split("x")
        self.width, self.height = int(dims[0]), int(dims[1])

        frame_size = self.width * self.height * 3 // 2  # YUV420p

        with open(self.path, "rb") as f:
            while True:
                data = f.read(frame_size)
                if len(data) < frame_size:
                    break
                components = DecorrelationStage.separate_yuv(data, self.width, self.height)
                rgb = components["rgb"]
                img = Image.fromarray(rgb, "RGB")
                self.frames.append(img)

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

        self.root.geometry("500x400")
        self.root.title("BildKomp")
        self.build_gui()

        if filename:
            self.load_video_file(filename, side="left")

    def build_gui(self):
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side="top", anchor="nw", fill="x")

        file_menu_btn = tk.Menubutton(self.top_frame, text="File", relief=tk.RAISED)
        file_menu = tk.Menu(file_menu_btn, tearoff=0)
        file_menu.add_command(label="Load Left Video", command=lambda: self.menu_load_file("left"))
        file_menu.add_command(label="Load Right Video", command=lambda: self.menu_load_file("right"))
        file_menu.add_command(label="Save as ...", command=self.menu_save_file)
        file_menu_btn.config(menu=file_menu)
        file_menu_btn.pack(side="left", padx=2, pady=2)

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
        self.frame_label_left = tk.Label(self.control_frame, text="Frame left side: -")
        self.frame_label_right = tk.Label(self.control_frame, text="Frame right side: -")
        
        self.info_frame = tk.Frame(self.main_frame, width=100)
        self.info_frame.pack(side="top", anchor="nw", padx=0)

        self.left_info_frame = tk.Frame(self.info_frame, width=300)
        self.left_info_frame.pack(side="left", anchor="nw")
        #tk.Label(self.left_info_frame, text="Info-Bereich\n(in Entwicklung)", anchor="center").pack(pady=0, padx=20)
        self.right_info_frame = tk.Frame(self.info_frame, width=30)
        self.right_info_frame.pack(side="left", anchor="nw", padx=0)
        #tk.Label(self.right_info_frame, text="VIDEO VON PSNR VISUALIZATION HERE\n", anchor="center").pack(pady=0, padx=0)

    def show_controls(self):
        self.control_frame.pack(pady=2, anchor="nw")
        self.start_btn.pack(side="left", padx=2)
        self.prev_btn.pack(side="left", padx=2)
        self.play_btn.pack(side="left", padx=2)
        self.next_btn.pack(side="left", padx=2)
        self.end_btn.pack(side="left", padx=2)
        self.frame_label_left.pack(side="left", padx=5)
        self.frame_label_right.pack(side="left", padx=5)

    def hide_controls(self):
        self.control_frame.pack_forget()

    def menu_load_file(self, side):
        path = filedialog.askopenfilename(
            initialdir=ROOT_DIR,
            filetypes=[("YUV or VID files", "*.yuv *.vid")]
        )
        if path:
            self.load_video_file(path, side=side)
            # if format == "yuv":
            #     self.load_yuv_file(path)
            # elif format == "vid":
            #     self.load_vid_file(path)


    def load_video_file(self, path, side):
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
            
        self.video_id += 2
        self.jump_to_start()
        self.display_frames()
        self.show_controls()

    def display_frames(self):
        if self.video_left and self.video_left.frames:
            frame = self.video_left.frames[self.frame_index_left]
            img = ImageTk.PhotoImage(frame)
            self.image_player_left.config(image=img, text="", compound=None)
            self.image_player_left.image = img
            
            frame_info_left = f"Frame left side: {self.frame_index_left if self.video_left else '-'} / "
            frame_info_left += f"{(len(self.video_left.frames) - 1) if self.video_left else '-'}"
            self.frame_label_left.config(text=frame_info_left)

        if self.video_right and self.video_right.frames:
            frame = self.video_right.frames[self.frame_index_right]
            img = ImageTk.PhotoImage(frame)
            self.image_player_right.config(image=img, text="", compound=None)
            self.image_player_right.image = img

            frame_info_right = f"Frame right side: {self.frame_index_right if self.video_right else '-'} / "
            frame_info_right += f"{(len(self.video_right.frames) - 1) if self.video_right else '-'}"
            self.frame_label_right.config(text=frame_info_right)

    def playback_loop(self):
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

    def menu_save_file(self):
        pass

if __name__ == "__main__":
    filename_arg = sys.argv[1] if len(sys.argv) >= 2 else None
    root = tk.Tk()
    app = Player(root, filename_arg)
    root.mainloop()
