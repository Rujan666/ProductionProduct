from video_camera import VideoCamera
import cv2
import tkinter as tk
from PIL import ImageTk, Image

def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2.imshow('Facial Expression Recognization',frame)
        key = cv2.waitKey(1)
        if key == 13:
            break
        elif key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def open_new_window(pred):
    root = tk.Tk()
    root.title("Result")
    text_label = tk.Label(root, text="You are feeling: ")
    text_label.pack()

    result_label = tk.Label(root, text=pred, font=("Arial", 22, "bold"))
    result_label.pack()

    image_path = "./avatars/" + pred.lower() + ".jpg"
    image = Image.open(image_path)
    image = image.resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.pack()

    button = tk.Button(root, text="Go to App", command=lambda: process_camera_feed(root))
    button.pack()

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate window position
    window_width = 500  # Set your desired window width
    window_height = 400  # Set your desired window height

    x_position = (screen_width // 2) - (window_width // 2)
    y_position = (screen_height // 2) - (window_height // 2)

    # Set window position
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    root.mainloop()


def process_camera_feed(root = None):
    camera = VideoCamera()
    gen(camera)

    pred = camera.get_pred()
    open_new_window(pred)

    if root != None:
        root.destroy()

process_camera_feed()

