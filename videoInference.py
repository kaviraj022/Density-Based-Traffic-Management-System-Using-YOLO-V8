from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Frame
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import threading
import time
import os
import copy

# Load the YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Define vehicle class IDs based on your data.yaml file
VEHICLE_CLASS_IDS = [0, 1, 2, 3]  # 'bicycle', 'bus', 'car', 'motorbike'
MAX_GREEN_TIME = 80  # Maximum green time in seconds

# Playback control flag
play_video = True

# Function to calculate green signal timing based on vehicle count
def calculate_green_time(vehicle_count):
    base_time = 10  # Base green signal duration in seconds
    time_per_vehicle = 5  # Additional time per vehicle
    calculated_time = max(base_time, vehicle_count * time_per_vehicle)
    return min(calculated_time, MAX_GREEN_TIME)

# Function to run YOLO detection on a frame
def detect_vehicles(frame):
    # Create a deep copy of the frame to keep the original raw frame intact
    annotated_frame = copy.deepcopy(frame)
    results = model(frame)
    vehicle_count = 0

    # Count vehicles and annotate the copy of the frame
    for box in results[0].boxes:
        class_id = int(box.cls[0])  # Get the class ID
        if class_id in VEHICLE_CLASS_IDS:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = model.names[class_id]
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    green_time = calculate_green_time(vehicle_count)
    return vehicle_count, green_time, annotated_frame

# Function to process video frames and save annotated video
def process_video(video_path):
    global play_video
    cap = cv2.VideoCapture(video_path)

    # Get total frames for the progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure the output directory exists
    output_dir = r'D:\Mini Project\output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup video writer for saving the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = os.path.join(output_dir, 'output_annotated.mp4')  # Save as .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    current_frame = 0
    start_time = time.time()

    while cap.isOpened():
        if not play_video:
            root.update()
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection on the frame (to create annotated frame)
        vehicle_count, green_time, annotated_frame = detect_vehicles(frame)
        current_frame += 1

        # Save the annotated frame
        out.write(annotated_frame)

        # Update progress bar
        progress = (current_frame / total_frames) * 100
        progress_bar['value'] = progress

        # Convert the raw frame and annotated frame for display
        raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL images
        raw_image = Image.fromarray(raw_image)
        annotated_image = Image.fromarray(annotated_image)

        # Resize images for the UI
        raw_image.thumbnail((300, 300))
        annotated_image.thumbnail((300, 300))

        # Convert to Tkinter-compatible images
        raw_tk_image = ImageTk.PhotoImage(raw_image)
        annotated_tk_image = ImageTk.PhotoImage(annotated_image)

        # Display raw video frame (left side)
        original_label.config(image=raw_tk_image)
        original_label.image = raw_tk_image

        # Display annotated video frame (right side)
        annotated_label.config(image=annotated_tk_image)
        annotated_label.image = annotated_tk_image

        # Update results
        result_label.config(
            text=f"Vehicle Count: {vehicle_count}\nGreen Signal Time: {green_time} seconds"
        )
        root.update_idletasks()
        root.update()

    cap.release()
    out.release()

# Function to open file dialog for video
def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not file_path:
        return

    # Run video processing in a separate thread to avoid blocking the UI
    threading.Thread(target=process_video, args=(file_path,)).start()

# Function to open file dialog for image
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    frame = cv2.imread(file_path)
    if frame is None:
        print("Error: Could not load image.")
        return

    # Detect vehicles on the image
    vehicle_count, green_time, annotated_frame = detect_vehicles(frame)

    # Convert the raw and annotated frames for display
    raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL images
    raw_image = Image.fromarray(raw_image)
    annotated_image = Image.fromarray(annotated_image)

    # Resize images for the UI
    raw_image.thumbnail((300, 300))
    annotated_image.thumbnail((300, 300))

    # Convert to Tkinter-compatible images
    raw_tk_image = ImageTk.PhotoImage(raw_image)
    annotated_tk_image = ImageTk.PhotoImage(annotated_image)

    # Display raw image (left side)
    original_label.config(image=raw_tk_image)
    original_label.image = raw_tk_image

    # Display annotated image (right side)
    annotated_label.config(image=annotated_tk_image)
    annotated_label.image = annotated_tk_image

    # Update results
    result_label.config(text=f"Vehicle Count: {vehicle_count}\nGreen Signal Time: {green_time} seconds")

# Function to toggle play/pause
def toggle_play_pause():
    global play_video
    play_video = not play_video
    play_button.config(text="Play" if not play_video else "Pause")

# Create the Tkinter window
root = tk.Tk()
root.title("Traffic Density Detection")
root.geometry("900x600")

# Buttons for upload options
upload_image_button = tk.Button(root, text="Upload Image", command=open_image)
upload_image_button.pack(pady=5)

upload_video_button = tk.Button(root, text="Upload Video", command=open_video)
upload_video_button.pack(pady=5)

# Frame for video displays
video_frame = Frame(root)
video_frame.pack(pady=10)

# Original video/image title and label
original_title = Label(video_frame, text="Original Frame", font=("Helvetica", 12))
original_title.grid(row=0, column=0, padx=10)

original_label = Label(video_frame)
original_label.grid(row=1, column=0, padx=10)

# Annotated video/image title and label
annotated_title = Label(video_frame, text="Annotated Frame", font=("Helvetica", 12))
annotated_title.grid(row=0, column=1, padx=10)

annotated_label = Label(video_frame)
annotated_label.grid(row=1, column=1, padx=10)

# Display progress bar
progress_bar = Progressbar(root, orient='horizontal', length=800, mode='determinate')
progress_bar.pack(pady=10)

# Display results
result_label = Label(root, text="Vehicle Count: \nGreen Signal Time: ", font=("Helvetica", 12))
result_label.pack(pady=10)

# Play/Pause button
play_button = tk.Button(root, text="Pause", command=toggle_play_pause)
play_button.pack(pady=5)

# Exit button
exit_button = tk.Button(root, text="Exit", command=root.destroy)
exit_button.pack(pady=10)

# Run the application
root.mainloop()
