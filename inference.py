from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk

# Load the YOLO model
model = YOLO('runs/detect/train/weights/best.pt')  

# Define vehicle class IDs based on your data.yaml file
VEHICLE_CLASS_IDS = [0, 1, 2, 3]  # 'bicycle', 'bus', 'car', 'motorbike'
MAX_GREEN_TIME = 80  # Maximum green time in seconds

# Function to calculate green signal timing based on vehicle count
def calculate_green_time(vehicle_count):
    base_time = 10  # Base green signal duration in seconds
    time_per_vehicle = 2  # Additional time per vehicle
    calculated_time = max(10, vehicle_count * time_per_vehicle)
    return min(calculated_time, MAX_GREEN_TIME)

# Function to run YOLO detection on the uploaded image
def detect_vehicles(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        return None, None
    
    results = model(frame)
    vehicle_count = 0

    # Count vehicles and annotate image
    for box in results[0].boxes:
        class_id = int(box.cls[0])  # Get the class ID
        if class_id in VEHICLE_CLASS_IDS:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = model.names[class_id]
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    green_time = calculate_green_time(vehicle_count)
    return vehicle_count, green_time, frame

# Function to open file dialog and display results
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Run detection
    vehicle_count, green_time, annotated_frame = detect_vehicles(file_path)
    if annotated_frame is None:
        return
    
    # Convert the image to Tkinter format and display it
    annotated_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(annotated_image)
    annotated_image.thumbnail((400, 400))
    tk_image = ImageTk.PhotoImage(annotated_image)
    
    image_label.config(image=tk_image)
    image_label.image = tk_image  # Keep a reference to avoid garbage collection

    # Update results
    result_label.config(text=f"Vehicle Count: {vehicle_count}\nGreen Signal Time: {green_time} seconds")

# Create the Tkinter window
root = tk.Tk()
root.title("Traffic Density Detection")
root.geometry("500x600")

# Upload button
upload_button = tk.Button(root, text="Upload Image", command=open_image)
upload_button.pack(pady=20)

# Display uploaded and processed image
image_label = Label(root)
image_label.pack(pady=10)

# Display results
result_label = Label(root, text="Vehicle Count: \nGreen Signal Time: ", font=("Helvetica", 12))
result_label.pack(pady=10)

# Run the application
root.mainloop()
