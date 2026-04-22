import cv2
import threading
import queue
import numpy as np
import math
from ultralytics import YOLOv10
from paddleocr import PaddleOCR
from datetime import datetime as dateTime
import re
import os
import mysql.connector
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="aslan",
    password="12345",
    database="vehicle_management"
)
cursor = db.cursor(buffered=True)
print("Connected to MySQL database 🚀")

# Video source

#rtsp_url = "rtsp:localhost:8554/mystream"
# cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture("../Resources/IPRV2.mp4")
if not cap.isOpened():
    print("Error: Could not open video file! Exiting...")
    exit()

# Models initialization
model = YOLOv10("weights/best.pt")
ocr = PaddleOCR(
    use_angle_cls=True, 
    use_gpu=False,
    rec_model_dir="../paddleocr/rec_model/")

# License plate pattern validation
plate_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{2}\d{4}$')
VALID_STATE_CODES = {"AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", 
                     "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"}

# Queues for threading
frame_queue = queue.Queue(maxsize=3)
yolo_queue = queue.Queue(maxsize=3)
ocr_queue = queue.Queue(maxsize=3)
db_queue = queue.Queue()

# Set to track processed license plates to avoid duplicates in quick succession
processed_plates = set()
last_plate_time = {}  # Track when a plate was last recorded
min_reentry_interval = 60  # Minimum seconds before considering the same plate as a new entry
exit_flag = False

def draw_bbox_with_label(frame, x1, y1, x2, y2, text, confidence):
    bbox_color = (0, 255, 0) if text else (0, 0, 255)  # Green = valid, Red = invalid
    conf_color = (0, 255, 0) if confidence >= 80 else (0, 255, 255) if confidence >= 50 else (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 3, cv2.LINE_AA)

    label = f"{text if text else 'Unknown'} ({confidence}%)"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w + 10, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)

def save_to_mysql(license_plate):
    """Record a vehicle entry in the MySQL database"""
    global db, cursor  # Access the global db and cursor variables
    current_time = dateTime.now()
    
    # Check if this plate was processed recently to avoid duplicate entries
    if license_plate in last_plate_time:
        time_diff = (current_time - last_plate_time[license_plate]).total_seconds()
        if time_diff < min_reentry_interval:
            return False  # Skip this entry as it's too soon after the previous one
    
    try:
        # Check if the vehicle exists in the vehicles table
        cursor.execute("SELECT COUNT(*) FROM vehicles WHERE number_plate = %s", (license_plate,))
        vehicle_exists = cursor.fetchone()[0] > 0
        
        if vehicle_exists:
            # If vehicle exists, log the entry in entry_logs table
            cursor.execute(
                """
                INSERT INTO entry_logs (number_plate, entry_time) 
                VALUES (%s, %s)
                """,
                (license_plate, current_time)
            )
            db.commit()
            print(f"✅ Recorded entry for known vehicle: {license_plate}")
        else:
            # If vehicle doesn't exist, log it in unknown_vehicles table
            # First, check if the unknown_vehicles table exists
            try:
                cursor.execute(
                    """
                    INSERT INTO unknown_vehicles (number_plate, entry_time) 
                    VALUES (%s, %s)
                    """,
                    (license_plate, current_time)
                )
                db.commit()
                print(f"⚠️ Recorded unknown vehicle: {license_plate}")
            except mysql.connector.Error as err:
                if "Table 'vehicle_management.unknown_vehicles' doesn't exist" in str(err):
                    # Create the table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE unknown_vehicles (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            number_plate VARCHAR(15) NOT NULL,
                            detection_time DATETIME NOT NULL,
                            processed BOOLEAN DEFAULT FALSE
                        )
                    """)
                    db.commit()
                    print("Created unknown_vehicles table")
                    
                    # Try inserting again
                    cursor.execute(
                        """
                        INSERT INTO unknown_vehicles (number_plate, detection_time) 
                        VALUES (%s, %s)
                        """,
                        (license_plate, current_time)
                    )
                    db.commit()
                    print(f"⚠️ Recorded unknown vehicle: {license_plate}")
                else:
                    raise
        
        last_plate_time[license_plate] = current_time
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        # Try to reconnect if connection is lost
        try:
            if not db.is_connected():
                db.reconnect()
                cursor = db.cursor(buffered=True)
                print("Reconnected to database")
        except:
            print("Failed to reconnect to database")
        return False

def video_thread():
    global exit_flag
    print("[VIDEO] Thread started...")
    while not exit_flag:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def yolo_thread():
    global exit_flag
    print("[YOLO] Thread started...")
    while not exit_flag:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        if frame is None:
            yolo_queue.put(None)
            break

        results = model.predict(frame, conf=0.25)
        detections = [(frame, *map(int, box.xyxy[0]), math.ceil(box.conf[0] * 100)) for result in results for box in result.boxes]
        if not yolo_queue.full():
            yolo_queue.put((frame, detections))

def ocr_thread():
    global exit_flag
    print("[OCR] Thread started...")
    while not exit_flag:
        try:
            frame, detections = yolo_queue.get(timeout=1)
        except queue.Empty:
            continue
        if frame is None:
            ocr_queue.put(None)
            break

        for frame, x1, y1, x2, y2, confidence in detections:
            cropped = frame[y1:y2, x1:x2]
            result = ocr.ocr(cv2.resize(cropped, (150, 50)), det=False, rec=True, cls=False)

            text = ""
            for r in result:
                if int(r[0][1] * 100) > 80:
                    text = re.sub(r'[^A-Z0-9]', '', r[0][0])

            # Validate the license plate format
            if plate_pattern.match(text) and text[:2] in VALID_STATE_CODES:
                # Log the entry to MySQL database
                if text not in processed_plates:
                    if save_to_mysql(text):
                        processed_plates.add(text)

            draw_bbox_with_label(frame, x1, y1, x2, y2, text, confidence)

        if not ocr_queue.full():
            ocr_queue.put(frame)

def database_thread():
    global exit_flag
    print("[DATABASE] Thread started...")
    while not exit_flag:
        try:
            frame = ocr_queue.get(timeout=1)
        except queue.Empty:
            continue
        if frame is None:
            break

        # Clear processed plates every minute to allow re-detection
        # of the same plate after sufficient time has passed
        if len(processed_plates) > 0 and time.time() % 60 < 1:
            processed_plates.clear()

        if not db_queue.full():
            db_queue.put(frame)

# Initialize threads
threads = [
    threading.Thread(target=video_thread),
    threading.Thread(target=yolo_thread),
    threading.Thread(target=ocr_thread),
    threading.Thread(target=database_thread)
]

# Start all threads
for t in threads:
    t.start()

# Main loop for displaying video
while True:
    try:
        frame = db_queue.get(timeout=1)
        if frame is None:
            break

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[EXIT] 'Q' pressed! Stopping all threads...")
            exit_flag = True
            break
    except queue.Empty:
        continue

# Clean up queues
for q in [frame_queue, yolo_queue, ocr_queue, db_queue]:
    while not q.empty():
        q.get()
    q.put(None)

# Wait for all threads to finish
for t in threads:
    t.join()

# Close database connection
cursor.close()
db.close()

# Clean up OpenCV windows
cv2.destroyAllWindows()
for _ in range(5):
    cv2.waitKey(1)

print("[EXIT] All threads stopped successfully. Exiting program.")