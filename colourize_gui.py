import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
from PIL import Image, ImageTk
# Paths to load the model
DIR = r"C:\Users\arjk2\OneDrive\Documents\Colorize"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the Model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image...")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    return colorized

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = Image.open(file_path)
        original_image = original_image.resize((300, 300))
        original_photo = ImageTk.PhotoImage(original_image)
        original_label.config(image=original_photo)
        original_label.image = original_photo
        original_label.file_path = file_path

def colorize():
    if not hasattr(original_label, 'file_path'):
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    file_path = original_label.file_path
    colorized_image = colorize_image(file_path)

    colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)
    colorized_image = Image.fromarray(colorized_image)
    colorized_image = colorized_image.resize((300, 300))
    colorized_photo = ImageTk.PhotoImage(colorized_image)

    colorized_label.config(image=colorized_photo)
    colorized_label.image = colorized_photo

def create_gradient_image():
    # Create a gradient image
    width, height = 650, 400
    gradient = Image.new('RGB', (width, height), color='white')
    for y in range(height):
        for x in range(width):
            r = int((x / width) * 0 + (y / height) * 0)
            g = int((x / width) * 255 + (y / height) * 255)
            b = int((x / width) * 255 + (y / height) * 255)
            gradient.putpixel((x, y), (r, g, b))
    return gradient

# Initialize the Tkinter window
root = tk.Tk()
root.title("Image Colorizer")
root.geometry("650x400")

# Create and set the gradient background
gradient_image = create_gradient_image()
gradient_photo = ImageTk.PhotoImage(gradient_image)

background_label = Label(root, image=gradient_photo)
background_label.place(relwidth=1, relheight=1)

# Add a title label
title_label = Label(root, text="Image Colorizer", font=("Bahnschrift SemiBold SemiConden", 24), bg="#FFFF00")
title_label.pack(pady=10)
3
# Create and place the buttons and labels with padding
button_frame = tk.Frame(root, bg="#999090")
button_frame.pack(pady=20)

upload_button = Button(button_frame, text="Upload Image", command=upload_image, font=("Arial", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
upload_button.pack(side="left", padx=20)

colorize_button = Button(button_frame, text="Colorize", command=colorize, font=("Arial", 14), bg="#2196F3", fg="white", padx=10, pady=5)
colorize_button.pack(side="right", padx=20)

image_frame = tk.Frame(root, bg="#D3D3D3")
image_frame.pack(pady=10)

original_label = Label(image_frame, bg="white", width=300, height=300)
original_label.pack(side="left", padx=10)

colorized_label = Label(image_frame, bg="white", width=300, height=300)
colorized_label.pack(side="right", padx=10)
root.mainloop()
