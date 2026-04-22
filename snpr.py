import cv2
from ultralytics import YOLOv10
import numpy as np
import math
from paddleocr import PaddleOCR
from datetime import datetime as dateTime
import re
import os
import json
import sqlite3
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialize model and OCR
cap = cv2.VideoCapture("../Resources/IPRV2.mp4")
if not cap.isOpened():
    print("Error: Could not open video stream! Exiting...")
    exit()
model = YOLOv10("weights/best.pt")

frame_skip = 1 
className = ['license_plate']
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, rec_model_dir="../paddleocr/rec_model/")

# Regex pattern for Indian license plates
plate_pattern = re.compile(r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{2}\s?\d{4}$')
VALID_STATE_CODES = {"AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", 
                     "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"}

# Shared queues for multi-threading
frame_queue = queue.Queue()
result_queue = queue.Queue()

def convert_to_grayscale(image):
    """ Convert image to grayscale with adaptive thresholding """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed

def paddle_ocr(frame, x1, y1, x2, y2):
    """ Perform OCR on cropped license plate """
    cropped = frame[y1:y2, x1:x2]
    processed = convert_to_grayscale(cropped)
    cropped_resized = cv2.resize(processed, (150, 50))   
    result = ocr.ocr(cropped_resized, det=False, rec=True, cls=False)

    text = ""
    for r in result:
        scores = r[0][1] if not np.isnan(r[0][1]) else 0
        scores = int(scores + 100)
        if scores > 60:
            text = r[0][0]

    text = re.sub(r'\W', '', text)
    if plate_pattern.match(text):
        state_code = text[:2] 
        if state_code in VALID_STATE_CODES:
            print(f"✅ Valid Plate Detected: {text}")
            return text
        else:
            print(f"❌ Invalid State Code: {text} (Ignoring)")
            return None
    else:
        print(f"❌ Invalid Plate Format: {text} (Ignoring)")
        return None

def save_json(license_plates, startTime, endTime):
    """ Save license plates to JSON """
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plates": list(license_plates)
    }

    os.makedirs("json", exist_ok=True)
    file_path = f"json/output_{dateTime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(file_path, "w") as f:
        json.dump(interval_data, f, indent=2)
    print(f"✅ JSON saved: {file_path}")

def save_to_db(license_plates, start_time, end_time):
    """ Save detected plates to SQLite database """
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('SELECT COUNT(*) FROM License_Plates WHERE license_plate = ?', (plate,))
        if cursor.fetchone()[0] > 0:
            continue 
        cursor.execute('INSERT INTO License_Plates(start_time, end_time, license_plate) VALUES(?, ?, ?)',
                       (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()
    print("✅ Database updated.")

def detect_license_plate():
    """ Run YOLO detection in a separate thread """
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        count += 1
        if count % frame_skip != 0:
            continue  

        print(f"Processing Frame: {count}")
        results = model.predict(frame, conf=0.25)
        frame_queue.put((frame, results))  

def process_frames():
    """ Process frames and perform OCR in parallel """
    license_plates = set()
    startTime = dateTime.now()

    while True:
        frame, results = frame_queue.get()
        currentTime = dateTime.now()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_text = paddle_ocr(frame, x1, y1, x2, y2)

                if plate_text:
                    license_plates.add(plate_text)

        if (currentTime - startTime).seconds >= 30:
            endTime = currentTime
            result_queue.put((license_plates, startTime, endTime))
            startTime = currentTime
            license_plates.clear()

def save_results():
    """ Save data to JSON and DB asynchronously """
    while True:
        license_plates, startTime, endTime = result_queue.get()
        save_json(license_plates, startTime, endTime)
        save_to_db(license_plates, startTime, endTime)

def display_video():
    """ Display processed frames """
    while True:
        if not frame_queue.empty():
            frame, _ = frame_queue.get()
            cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(detect_license_plate)
        executor.submit(process_frames)
        executor.submit(save_results)
        executor.submit(display_video)

    cap.release()
    cv2.destroyAllWindows()