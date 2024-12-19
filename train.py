from ultralytics import YOLO

# Load the model 
model = YOLO('yolov8n.pt')  

# Train the model
model.train(data='D:\Mini Project\data.yaml', epochs=50, imgsz=640)  