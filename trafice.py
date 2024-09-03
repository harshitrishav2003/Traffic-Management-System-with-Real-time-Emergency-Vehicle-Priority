import tkinter as tk
from tkinter import filedialog, ttk, messagebox  # Import messagebox module
from keras.models import load_model
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import time

# Global variables for traffic light colors and green times
traffic_light_colors = {"red": "#FF0000", "orange": "#FFA500", "green": "#00FF00"}
green_times = {"empty road": 2, "normal traffic": 3, "heavy traffic": 4, "emergency vehicle ": 8}
current_object = None

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

def classify_image(image):
    global current_object
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # Display the input image
    input_img = ImageTk.PhotoImage(image)
    input_img_label.config(image=input_img)
    input_img_label.image = input_img

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    # Check if the index is within the range of class_names
    if index < len(class_names):
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # Update the result label with the predicted class name
        result_label.config(text=f"Predicted Class: {class_name}")
        
        # Check if the confidence score is greater than 0.98
        if confidence_score > 0.90:
            confidence_label.config(text=f"Confidence Score: {confidence_score:.4f}")
            confidence_label.pack()
            result_label.pack()  # Display the Predicted Class label
            # Update the current detected object
            current_object = class_name.lower()
            update_traffic_light(current_object)
            start_timer(green_times.get(current_object, 0))  # Use get() to handle missing keys
        else:
            confidence_label.pack_forget()  # Hide the Confidence Score label
            result_label.pack_forget()  # Hide the Predicted Class label
            current_object = None
            
    else:
        result_label.config(text="Not Detected")
        confidence_label.pack_forget()  # Hide the Confidence Score label
        result_label.pack()  # Display the Predicted Class label

def update_traffic_light(object_name):
    color = traffic_light_colors["red"]  # Default to red
    if object_name == "empty road":
        color = traffic_light_colors["green"]
    elif object_name == "normal traffic":
        color = traffic_light_colors["orange"]
    elif object_name == "heavy traffic":
        color = traffic_light_colors["green"]
    elif object_name == "emergency vehicle ":
        color = traffic_light_colors["green"]
    traffic_light_canvas.itemconfig(traffic_light_circle, fill=color)

def start_timer(seconds):
    update_timer_label(seconds)
    while seconds > 0:
        time.sleep(1)
        seconds -= 1
        update_timer_label(seconds)
        root.update()
    update_timer_label(0)

def update_timer_label(seconds):
    timer_label.config(text=f"Time remaining: {seconds} s")

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        classify_image(image)

def capture_from_webcam():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            classify_image(image)
            
            # Update GUI
            root.update_idletasks()
            root.update()
            
            # Delay for smooth streaming (adjust as needed)
            root.after(50)
        else:
            messagebox.showerror("Error", "Unable to read frame from webcam.")
            break
            
    cap.release()

# Create the main window
root = tk.Tk()
root.title("Traffic Light Control System")
root.geometry("600x600")

# Update the GUI colors
root.configure(bg="#F0F0F0")
style = ttk.Style()
style.theme_use("clam")  # You can try other themes like "default", "alt", etc.

# Create a frame for the title
title_frame = tk.Frame(root, bg="#F0F0F0")
title_frame.pack(pady=20)

title_label = tk.Label(title_frame, text="Traffic Light Control System", font=("Arial", 16, "bold"), bg="#F0F0F0")
title_label.pack()

# Create a frame for input image
input_frame = tk.Frame(root, bg="#E0E0E0")
input_frame.pack(pady=20)

input_label = tk.Label(input_frame, text="Input Image", font=("Arial", 14, "bold"), bg="#E0E0E0")
input_label.pack()

input_img_label = tk.Label(input_frame, bg="#E0E0E0")
input_img_label.pack()

# Create a frame for result
result_frame = tk.Frame(root, bg="#E0E0E0")
result_frame.pack(pady=20)

result_label = tk.Label(result_frame, text="", font=("Arial", 16), bg="#E0E0E0")
result_label.pack()

# Create a label for Confidence Score
confidence_label = tk.Label(root, text="", font=("Arial", 14), bg="#F0F0F0")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image, bg="#4CAF50", fg="white", font=("Arial", 12))
select_button.pack(pady=10)

# Create a button to capture image from webcam
webcam_button = tk.Button(root, text="Start Live Stream", command=capture_from_webcam, bg="#4CAF50", fg="white", font=("Arial", 12))
webcam_button.pack(pady=10)

# Create a traffic light
traffic_light_canvas = tk.Canvas(root, width=50, height=150)
traffic_light_canvas.pack()
traffic_light_canvas.create_rectangle(20, 20, 50, 130, fill="black")
traffic_light_circle = traffic_light_canvas.create_oval(25, 25, 45, 45, fill="red")

# Create a timer label
timer_label = tk.Label(root, text="Time remaining: 0 s", font=("Arial", 14))
timer_label.pack()

# Run the Tkinter 
root.mainloop()
