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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Video file input
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video file! Exiting...")
    exit()

# Load YOLOv10 model for license plate detection
model = YOLOv10("weights/best.pt")

count = 0
frame_skip = 1  # Process every 2nd frame
className = ['license_plate']

# Load PaddleOCR with the downloaded recognition model
ocr = PaddleOCR(
    use_angle_cls=True, 
    use_gpu=False,
    rec_model_dir="../paddleocr/rec_model/")

VALID_STATE_CODES = {"AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", 
                     "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB"}

plate_pattern = re.compile(r'^(?:AP|AR|AS|BR|CG|GA|GJ|HR|HP|JH|KA|KL|MP|MH|MN|ML|MZ|NL|OD|PB|RJ|SK|TN|TS|TR|UP|UK|WB)\d{1,2}[A-Z]{2}\d{4}$')

def convert_to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Apply adaptive thresholding for enhanced contrast
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return processed

def paddle_ocr(frame, x1, y1, x2, y2):
    """Extract text from the detected license plate using PaddleOCR."""
    cropped = frame[y1:y2, x1:x2]
    processed = convert_to_grayscale(cropped)
    cropped_resized = cv2.resize(processed, (150, 50))  # Resize for better OCR accuracy
    result = ocr.ocr(cropped_resized, det=False, rec=True, cls=False)

    text = ""
    for r in result:
        scores = r[0][1] if not np.isnan(r[0][1]) else 0
        scores = int(scores + 100)
        if scores > 80:  # Increased Confidence threshold
            text = r[0][0]

    text = re.sub(r'[^A-Z0-9]', '', text)  # Remove non-alphanumeric characters
    
    if plate_pattern.match(text):
        state_code = text[:2]  # Extract state code
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

startTime = dateTime.now()
license_plates = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    currentTime = dateTime.now()
    count += 1
    if count % frame_skip != 0:
        continue  

    print(f"Frame Number: {count}")
    results = model.predict(frame, conf=0.25)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate_text = paddle_ocr(frame, x1, y1, x2, y2)

            if plate_text:
                state_code = plate_text[:2]
                bbox_color = (0, 255, 0) if state_code in VALID_STATE_CODES else (0, 0, 255) # Green for valid, Red for invalid
                license_plates.add(plate_text)
            else:
                bbox_color = (0, 0, 255)  # Default to red if OCR fails

            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 3, cv2.LINE_AA)

            conf = math.ceil(box.conf[0] * 100)
            conf_color = (0, 255, 0) if conf >= 80 else (0, 255, 255) if conf >= 50 else (0, 0, 255)  # Green, Yellow, Red

            conf_text = f"Conf: {conf}%"
            conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            conf_w, conf_h = conf_size
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + conf_w + 10, y2 + conf_h + 10), (0, 0, 0), -1)  # Black background
            cv2.putText(frame, conf_text, (x1 + 5, y2 + conf_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)

            if plate_text:
                text_size, _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_w, text_h = text_size
                text_x1, text_y1 = x1, y1 - text_h - 10
                cv2.rectangle(frame, (text_x1, text_y1), (text_x1 + text_w + 10, y1 - 2), (0, 0, 0), -1)  # Black background
                cv2.putText(frame, plate_text, (text_x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) 

    if (currentTime - startTime).seconds >= 30:
        endTime = currentTime
        save_json(license_plates, startTime, endTime)
        save_to_db(license_plates, startTime, endTime)
        startTime = currentTime
        license_plates.clear()

    cv2.imshow("Video", frame)
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
