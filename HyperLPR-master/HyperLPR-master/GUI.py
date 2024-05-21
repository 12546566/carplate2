import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
from main import recognize_from_photo, recognize_from_video, recognize_from_camera
import HyperLPRLite as pr

class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别")

        # 设置窗口为全屏
        self.root.state('zoomed')

        # 左侧显示原始图像的框架
        self.image_frame = Frame(root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.original_image_label = Label(self.image_frame, text="原图：")
        self.original_image_label.pack(anchor="nw")

        self.image_label = Label(self.image_frame)
        self.image_label.pack()

        # 右侧显示识别结果和车牌颜色的框架
        self.result_frame = Frame(root)
        self.result_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.result_label = Label(self.result_frame, text="车牌识别结果：", anchor="nw", justify=tk.LEFT)
        self.result_label.pack(anchor="nw")

        self.color_label = Label(self.result_frame, text="车牌颜色：", anchor="nw", justify=tk.LEFT)
        self.color_label.pack(anchor="nw")

        # 底部按钮框架
        self.button_frame = Frame(root)
        self.button_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.select_image_button = Button(self.button_frame, text="来自图片", command=self.load_image)
        self.select_image_button.pack(fill='x')

        self.camera_button = Button(self.button_frame, text="来自摄像头", command=self.use_camera)
        self.camera_button.pack(fill='x')

        self.recognize_button = Button(self.button_frame, text="展示图片", command=self.recognize_plate)
        self.recognize_button.pack(fill='x')

        self.model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
        self.file_path = None
        self.source_type = None

    def load_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.source_type = 'image'
            image = Image.open(self.file_path)
            image.thumbnail((800, 800))
            self.image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image)

    def use_camera(self):
        self.source_type = 'camera'
        self.result_label.config(text="Using camera for real-time recognition.")

    def recognize_plate(self):
        if self.source_type == 'image' and self.file_path:
            results = recognize_from_photo(self.file_path, self.model)
            self.display_results(results)
        elif self.source_type == 'camera':
            recognize_from_camera(self.model)
        else:
            self.result_label.config(text="No valid source selected.")

    def display_results(self, results):
        if results:
            plate_results = "\n".join([f"Plate: {res[0]}, Confidence: {res[1]}" for res in results])
            color_results = "\n".join([f"Color: {res[3]}" for res in results])
        else:
            plate_results = "No plate detected."
            color_results = "No color detected."
        self.result_label.config(text=f"车牌识别结果：\n{plate_results}")
        self.color_label.config(text=f"车牌颜色：\n{color_results}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
