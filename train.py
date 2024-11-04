from ultralytics import YOLO

# Load the model (you can start with a pretrained model)
model = YOLO('yolov8n.pt')  # You can choose other versions based on your requirements

# Train the model
model.train(data='D:\Mini Project\data.yaml', epochs=50, imgsz=640)  # Adjust epochs and image size as needed
