from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import time
import threading
import base64
import numpy as np
import copy
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_GREEN_TIME = 30  # Maximum 30 seconds per lane
VEHICLE_CLASS_IDS = [0, 1, 2, 3]  # 'bicycle', 'bus', 'car', 'motorbike'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLO model lazily
model = None

def get_model():
    """Get or load the YOLO model"""
    global model
    if model is None:
        import torch
        model_path = 'runs/detect/train/weights/best.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
        
        # Patch torch.load to use weights_only=False for compatibility with PyTorch 2.6+
        original_load = torch.load
        def patched_load(*args, **kwargs):
            # Set weights_only=False for compatibility with older model files
            if 'weights_only' in kwargs and kwargs['weights_only']:
                kwargs['weights_only'] = False
            elif 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = patched_load
        try:
            model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            torch.load = original_load
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
        finally:
            # Restore original torch.load
            torch.load = original_load
    return model

# Global state for traffic light system
traffic_state = {
    'current_lane': 0,  # 0-3 for lanes 1-4
    'lane_start_time': None,
    'lane_videos': [None, None, None, None],
    'lane_caps': [None, None, None, None],
    'lane_frames': [None, None, None, None],  # Store current frames for each lane
    'lane_annotated_frames': [None, None, None, None],  # Store annotated frames
    'lane_vehicle_counts': [0, 0, 0, 0],  # Cache vehicle counts
    'lane_last_detection_time': [0, 0, 0, 0],  # Last detection time for each lane
    'lane_positions': [0, 0, 0, 0],  # Store frame positions for each lane
    'is_running': False,
    'cycle_complete': False,
    'last_vehicle_check_time': None  # Last time we checked for vehicles on active lane
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_vehicles(frame):
    """Detect vehicles in a frame and return count and annotated frame"""
    annotated_frame = copy.deepcopy(frame)
    yolo_model = get_model()
    results = yolo_model(frame)
    vehicle_count = 0

    # Count vehicles and annotate the frame
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id in VEHICLE_CLASS_IDS:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = yolo_model.names[class_id]
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vehicle_count, annotated_frame

def get_frame_from_video(cap):
    """Get a frame from video capture"""
    if cap is None or not cap.isOpened():
        return None, None
    ret, frame = cap.read()
    if not ret:
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return ret, frame

def process_traffic_light_logic():
    """Main traffic light logic - runs in background thread"""
    global traffic_state
    
    while traffic_state['is_running'] and not traffic_state['cycle_complete']:
        current_lane = traffic_state['current_lane']
        
        # Check if we've completed all lanes
        if current_lane >= 4:
            traffic_state['cycle_complete'] = True
            traffic_state['is_running'] = False
            break
        
        # Initialize lane start time
        if traffic_state['lane_start_time'] is None:
            print(f"Starting lane {current_lane + 1}")
            traffic_state['lane_start_time'] = time.time()
            traffic_state['last_vehicle_check_time'] = time.time()
            # Reset video to beginning when starting a new lane
            cap = traffic_state['lane_caps'][current_lane]
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                traffic_state['lane_positions'][current_lane] = 0
                # Read first frame and do initial detection
                ret, frame = cap.read()
                if ret:
                    traffic_state['lane_frames'][current_lane] = frame.copy()
                    # Do initial vehicle detection
                    vehicle_count, annotated_frame = detect_vehicles(frame)
                    traffic_state['lane_vehicle_counts'][current_lane] = vehicle_count
                    traffic_state['lane_annotated_frames'][current_lane] = annotated_frame
                    traffic_state['lane_last_detection_time'][current_lane] = time.time()
                    print(f"Lane {current_lane + 1} initial detection: {vehicle_count} vehicles")
        
        # Get video capture for current lane
        cap = traffic_state['lane_caps'][current_lane]
        if cap is None:
            # No video for this lane, skip to next
            print(f"No video for lane {current_lane + 1}, skipping to next lane")
            traffic_state['current_lane'] += 1
            traffic_state['lane_start_time'] = None
            time.sleep(0.5)
            continue
        
        # Check elapsed time
        elapsed_time = time.time() - traffic_state['lane_start_time']
        
        # Check if maximum time elapsed
        if elapsed_time >= MAX_GREEN_TIME:
            # Switch to next lane
            print(f"Lane {current_lane + 1} reached max time ({MAX_GREEN_TIME}s), switching to lane {current_lane + 2}")
            traffic_state['current_lane'] += 1
            traffic_state['lane_start_time'] = None
            time.sleep(1)  # Brief pause between lane switches
            continue
        
        # Advance video for active lane only (play at normal speed)
        ret, frame = cap.read()
        if not ret:
            # Video ended, reset to beginning and loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        if ret:
            # Store the frame for this lane
            traffic_state['lane_frames'][current_lane] = frame.copy()
            traffic_state['lane_positions'][current_lane] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Check for vehicles every 2 seconds (to reduce computation)
            if traffic_state['last_vehicle_check_time'] is None:
                traffic_state['last_vehicle_check_time'] = time.time()
            
            time_since_last_check = time.time() - traffic_state['last_vehicle_check_time']
            if time_since_last_check >= 2.0:
                vehicle_count, annotated_frame = detect_vehicles(frame)
                traffic_state['lane_vehicle_counts'][current_lane] = vehicle_count
                traffic_state['lane_annotated_frames'][current_lane] = annotated_frame
                traffic_state['last_vehicle_check_time'] = time.time()
                traffic_state['lane_last_detection_time'][current_lane] = time.time()
                
                print(f"Lane {current_lane + 1}: {vehicle_count} vehicles detected, elapsed: {elapsed_time:.1f}s")
                
                # Switch lane if no vehicles detected (but only after checking for at least 2 seconds)
                if vehicle_count == 0 and elapsed_time >= 2.0:
                    # Switch to next lane
                    print(f"Lane {current_lane + 1} has no vehicles after {elapsed_time:.1f}s, switching to lane {current_lane + 2}")
                    traffic_state['current_lane'] += 1
                    traffic_state['lane_start_time'] = None
                    time.sleep(1)  # Brief pause between lane switches
                    continue
        else:
            # Could not read frame, skip to next lane
            print(f"Could not read frame from lane {current_lane + 1}, switching to next lane")
            traffic_state['current_lane'] += 1
            traffic_state['lane_start_time'] = None
            time.sleep(0.5)
            continue
        
        time.sleep(0.033)  # ~30 FPS for video playback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video uploads for each lane"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    lane = request.form.get('lane')
    if not lane or lane not in ['lane1', 'lane2', 'lane3', 'lane4']:
        return jsonify({'error': 'Invalid lane specified'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{lane}_{filename}")
        file.save(filepath)
        
        # Store video path
        lane_index = int(lane[-1]) - 1
        traffic_state['lane_videos'][lane_index] = filepath
        
        # Open video capture
        cap = cv2.VideoCapture(filepath)
        traffic_state['lane_caps'][lane_index] = cap
        
        # Read first frame
        ret, frame = cap.read()
        if ret:
            traffic_state['lane_frames'][lane_index] = frame.copy()
            traffic_state['lane_positions'][lane_index] = 0
            # Initialize detection cache
            vehicle_count, annotated_frame = detect_vehicles(frame)
            traffic_state['lane_vehicle_counts'][lane_index] = vehicle_count
            traffic_state['lane_annotated_frames'][lane_index] = annotated_frame
            traffic_state['lane_last_detection_time'][lane_index] = time.time()
        
        return jsonify({'message': f'Video uploaded successfully for {lane}', 'filepath': filepath})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/start', methods=['POST'])
def start_system():
    """Start the traffic light system"""
    global traffic_state
    
    # Check if all lanes have videos
    if not all(traffic_state['lane_videos']):
        return jsonify({'error': 'Please upload videos for all 4 lanes'}), 400
    
    # Reset state
    traffic_state['current_lane'] = 0  # Start with lane 1 (index 0)
    traffic_state['lane_start_time'] = None
    traffic_state['is_running'] = True
    traffic_state['cycle_complete'] = False
    traffic_state['last_vehicle_check_time'] = None
    
    # Reset all video captures to beginning
    for i, cap in enumerate(traffic_state['lane_caps']):
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Reset detection cache
            traffic_state['lane_vehicle_counts'][i] = 0
            traffic_state['lane_last_detection_time'][i] = 0
    
    # Start traffic light logic in background thread
    thread = threading.Thread(target=process_traffic_light_logic, daemon=True)
    thread.start()
    
    return jsonify({'message': 'Traffic light system started'})

@app.route('/stop', methods=['POST'])
def stop_system():
    """Stop the traffic light system"""
    global traffic_state
    traffic_state['is_running'] = False
    return jsonify({'message': 'Traffic light system stopped'})

@app.route('/status', methods=['GET'])
def get_status():
    """Get current traffic light status"""
    global traffic_state
    
    current_lane = traffic_state['current_lane']
    elapsed_time = 0
    if traffic_state['lane_start_time']:
        elapsed_time = time.time() - traffic_state['lane_start_time']
    
    # Get vehicle count from cache (no detection needed here)
    vehicle_count = 0
    if current_lane < 4:
        vehicle_count = traffic_state['lane_vehicle_counts'][current_lane]
    
    return jsonify({
        'current_lane': current_lane + 1,  # 1-4 instead of 0-3
        'elapsed_time': round(elapsed_time, 1),
        'vehicle_count': vehicle_count,
        'is_running': traffic_state['is_running'],
        'cycle_complete': traffic_state['cycle_complete']
    })

@app.route('/frame/<int:lane>', methods=['GET'])
def get_frame(lane):
    """Get current frame from a specific lane"""
    if lane < 1 or lane > 4:
        return jsonify({'error': 'Invalid lane number'}), 400
    
    lane_index = lane - 1
    cap = traffic_state['lane_caps'][lane_index]
    
    if cap is None:
        return jsonify({'error': 'No video for this lane'}), 400
    
    # Use stored frame if available (don't advance video for inactive lanes)
    frame = traffic_state['lane_frames'][lane_index]
    
    if frame is None:
        # Read first frame if we don't have one stored
        # This happens when video is first uploaded
        ret, frame = cap.read()
        if not ret:
            # Reset to beginning if video ended
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return jsonify({'error': 'Could not read frame'}), 400
        traffic_state['lane_frames'][lane_index] = frame.copy()
        # Initialize detection for first frame
        if traffic_state['lane_annotated_frames'][lane_index] is None:
            vehicle_count, annotated_frame = detect_vehicles(frame)
            traffic_state['lane_vehicle_counts'][lane_index] = vehicle_count
            traffic_state['lane_annotated_frames'][lane_index] = annotated_frame
            traffic_state['lane_last_detection_time'][lane_index] = time.time()
    
    # Use cached annotated frame if available and recent (within 3 seconds)
    annotated_frame = traffic_state['lane_annotated_frames'][lane_index]
    vehicle_count = traffic_state['lane_vehicle_counts'][lane_index]
    time_since_detection = time.time() - traffic_state['lane_last_detection_time'][lane_index]
    
    # Only re-detect if cache is stale (older than 3 seconds) or doesn't exist
    if annotated_frame is None or time_since_detection > 3.0:
        # Only detect if this is the active lane (to save computation)
        current_lane = traffic_state['current_lane'] + 1
        if current_lane == lane or annotated_frame is None:
            vehicle_count, annotated_frame = detect_vehicles(frame)
            traffic_state['lane_vehicle_counts'][lane_index] = vehicle_count
            traffic_state['lane_annotated_frames'][lane_index] = annotated_frame
            traffic_state['lane_last_detection_time'][lane_index] = time.time()
        else:
            # For inactive lanes, just use the raw frame with basic annotation
            annotated_frame = frame.copy()
    
    # Add lane indicator and traffic light status
    current_lane = traffic_state['current_lane'] + 1
    is_active = (traffic_state['is_running'] and current_lane == lane)
    
    # Draw traffic light status on frame
    light_color = (0, 255, 0) if is_active else (0, 0, 255)  # Green if active, red otherwise
    cv2.rectangle(annotated_frame, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f'Lane {lane}', (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f'Vehicles: {vehicle_count}', (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(annotated_frame, (220, 45), 15, light_color, -1)
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'frame': f'data:image/jpeg;base64,{frame_base64}',
        'vehicle_count': vehicle_count
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

