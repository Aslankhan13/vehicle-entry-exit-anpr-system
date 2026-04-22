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
import json
import sqlite3  

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cap = cv2.VideoCapture("../Resources/IPRV2.mp4")
if not cap.isOpened():
    print("Error: Could not open video file! Exiting...")
    exit()

model = YOLOv10("weights/best.pt")
ocr = PaddleOCR(
    use_angle_cls=True, 
    use_gpu=False,
    rec_model_dir="../paddleocr/rec_model/")

plate_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{2}\d{4}$')
VALID_STATE_CODES = {"AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", 
                     "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"}

frame_queue = queue.Queue(maxsize=5)  # Reduced maxsize to prevent overload
yolo_queue = queue.Queue(maxsize=5)
ocr_queue = queue.Queue(maxsize=5)
db_queue = queue.Queue()

license_plates = set()
startTime = dateTime.now()
exit_flag = False  

def draw_bbox_with_label(frame, x1, y1, x2, y2, text, confidence):
    bbox_color = (0, 255, 0) if text else (0, 0, 255)  # Green = valid, Red = invalid
    conf_color = (0, 255, 0) if confidence >= 80 else (0, 255, 255) if confidence >= 50 else (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 3, cv2.LINE_AA)

    label = f"{text if text else 'Unknown'} ({confidence}%)"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w + 10, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)

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

            if plate_pattern.match(text) and text[:2] in VALID_STATE_CODES:
                license_plates.add(text)

            draw_bbox_with_label(frame, x1, y1, x2, y2, text, confidence)

        if not ocr_queue.full():
            ocr_queue.put(frame)

def database_thread():
    global exit_flag, startTime
    print("[DATABASE] Thread started...")
    while not exit_flag:
        try:
            frame = ocr_queue.get(timeout=1)
        except queue.Empty:
            continue
        if frame is None:
            break

        currentTime = dateTime.now()
        if (currentTime - startTime).seconds >= 10:
            endTime = currentTime
            save_to_db(license_plates, startTime, endTime)
            save_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()

        if not db_queue.full():
            db_queue.put(frame)

def save_json(license_plates, startTime, endTime):
    """Save detected license plates to a JSON file."""
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plates": list(license_plates)
    }

    interval_file_path = "json/output_" + dateTime.now().strftime("%Y%m%d%H%M%S") + ".json"
    try:
        with open(interval_file_path, "w") as f:
            json.dump(interval_data, f, indent=2)
        print(f"✅ Created file: {interval_file_path}")
    except Exception as e:
        print(f"❌ Error writing to file {interval_file_path}: {e}")

    cummulative_file_path = "json/LicensePlateData.json"

    try:
        if os.path.exists(cummulative_file_path):
            with open(cummulative_file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        print("⚠️ Cumulative JSON file was corrupted. Resetting...")
                        existing_data = []
                except json.JSONDecodeError:
                    print("⚠️ Cumulative JSON file is empty or invalid. Resetting...")
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(interval_data)

        with open(cummulative_file_path, "w") as f:
            json.dump(existing_data, f, indent=2)
        print(f"✅ Updated cumulative file: {cummulative_file_path}")

    except Exception as e:
        print(f"❌ Error writing to cumulative file {cummulative_file_path}: {e}")

def save_to_db(license_plates, start_time, end_time):
    """Save detected license plates to an SQLite database."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    for plate in license_plates:
        cursor.execute('SELECT COUNT(*) FROM LicensePlates WHERE license_plate = ?', (plate,))
        if cursor.fetchone()[0] > 0:
            continue  # Skip if duplicate

        cursor.execute(
            '''
            INSERT INTO LicensePlates(start_time, end_time, license_plate) 
            VALUES(?, ?, ?)
            ''',
            (start_time.isoformat(), end_time.isoformat(), plate)
        )
    conn.commit()
    conn.close()


threads = [
    threading.Thread(target=video_thread),
    threading.Thread(target=yolo_thread),
    threading.Thread(target=ocr_thread),
    threading.Thread(target=database_thread)
]

for t in threads:
    t.start()

while True:
    try:
        frame = db_queue.get(timeout=1)
        if frame is None:
            break

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF  # Ensure correct key press detection
        if key == ord('q'):
            print("\n[EXIT] 'Q' pressed! Stopping all threads...")
            exit_flag = True  # Stop all threads
            break
    except queue.Empty:
        continue

# Force all queues to stop
while not frame_queue.empty():
    frame_queue.get()
while not yolo_queue.empty():
    yolo_queue.get()
while not ocr_queue.empty():
    ocr_queue.get()
while not db_queue.empty():
    db_queue.get()

frame_queue.put(None)
yolo_queue.put(None)
ocr_queue.put(None)
db_queue.put(None)

for t in threads:
    t.join()

cv2.destroyAllWindows()
for _ in range(5):  # Ensure OpenCV windows close
    cv2.waitKey(1)

print("[EXIT] All threads stopped successfully. Exiting program.")
