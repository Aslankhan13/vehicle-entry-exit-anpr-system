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
cap = cv2.VideoCapture("../Resources/IPR2.mkv")
if not cap.isOpened():
    print("Error: Could not open video file! Exiting...")
    exit()

model = YOLOv10("C:/Users/aslan/OneDrive/Desktop/VEEMS/yolov10/weights/best.pt")

count = 0
frame_skip = 4   # Process every 2nd frame (adjust as needed)

className = ['license_plate']

ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)

def paddle_ocr(frame, x1, y1, x2, y2):
    cropped = frame[y1:y2, x1:x2]
    cropped = cv2.resize(cropped, (150, 50))  # Resize to improve OCR speed
    result = ocr.ocr(cropped, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1] if not np.isnan(r[0][1]) else 0
        scores = int(scores + 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile(r'\W')
    text = pattern.sub('', text)
    text = text.replace("???", "").replace("O", "0").replace("粤", "")
    return str(text)

def save_json(license_plates, startTime, endTime):
    interval_data = {
        "Start Time":startTime.isoformat(),
        "End Time":endTime.isoformat(),
        "License Plates":list(license_plates)
    }
    interval_file_path = "json/output_" + dateTime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, "w") as f:
        json.dump(interval_data, f, indent=2)

    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    existing_data.append(interval_data)

    with open(cummulative_file_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    save_to_db(license_plates, startTime, endTime)  

def save_to_db(license_plates, start_time, end_time):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    for plate in license_plates:
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
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            classNameInt = int(box.cls[0])
            clsName = className[classNameInt]
            conf = math.ceil(box.conf[0] * 100) / 100
            
            label = paddle_ocr(frame, x1, y1, x2, y2)

            if label:
                license_plates.add(label)

            textSize = cv2.getTextSize(label, 0, 0.5, 2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if(currentTime - startTime).seconds >= 30:
        endTime = currentTime
        save_json(license_plates, startTime, endTime)
        startTime = currentTime
        license_plates.clear()

    cv2.imshow("Video", frame)
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()