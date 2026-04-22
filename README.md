# 🚗 Vehicle Entry–Exit Management System (ANPR Based)

An AI-powered, real-time **Automatic Number Plate Recognition (ANPR)** system built for secure facility monitoring. The system processes live RTSP video streams, detects and reads vehicle license plates, and automatically logs entry/exit records — eliminating manual vehicle tracking entirely.

---

## 📸 Demo

> _Add a screenshot or GIF of your system running here_

---

## 🧠 How It Works

1. Live video feed is captured via **RTSP stream**
2. **YOLOv10** detects license plates in each frame in real time
3. **PaddleOCR** extracts the plate text from detected regions
4. Plate data is validated and stored in the database with a timestamp
5. Entry/exit logs are accessible via the dashboard

---

## ✨ Features

- 🎥 Real-time RTSP video stream processing
- 🔍 License plate detection using YOLOv10
- 🔤 OCR-based plate text extraction using PaddleOCR
- 📋 Automated vehicle entry/exit logging
- ⚡ Optimized for edge deployment on NVIDIA Jetson Nano
- 🚫 False positive reduction via confidence threshold filtering
- 📊 Dashboard for monitoring and log management
- 🐳 Docker support for easy deployment

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv10 |
| OCR | PaddleOCR |
| Video Processing | OpenCV |
| GPU Acceleration | CUDA |
| Edge Deployment | NVIDIA Jetson Nano |
| Backend | Python |
| Database | MySQL |
| Containerization | Docker |

---

## 📁 Project Structure

```
ANPR system/
├── build/              # Build files
├── docker/             # Docker configuration
├── docs/               # Documentation
├── examples/           # Example scripts and demos
├── figures/            # Images and diagrams
├── json/               # JSON config files
├── logs/               # Log output files
├── runs/               # Model inference runs
├── tests/              # Test scripts
├── Train_0/            # Training dataset and configs
├── weights/            # Trained model weights
├── anpr.py             # Main ANPR pipeline
├── app.py              # Application entry point
├── database.py         # Database handler
├── main.py             # Main runner
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (or NVIDIA Jetson Nano)
- MySQL

### Steps

**1. Clone the repository:**
```bash
git clone https://github.com/Aslankhan13/vehicle-entry-exit-anpr-system.git
cd vehicle-entry-exit-anpr-system
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
pip install ultralytics
pip install paddlepaddle paddleocr
```

**4. Set up the database:**
```bash
# Import the provided SQL schema
mysql -u root -p < database.sql
```

**5. Configure your RTSP stream:**
Edit `main.py` and update the RTSP URL:
```python
RTSP_URL = "rtsp://your-camera-ip/stream"
```

**6. Run the system:**
```bash
python main.py
```

---

## 🚀 Docker Deployment

```bash
docker build -t anpr-system .
docker run --gpus all anpr-system
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Detection Model | YOLOv10 |
| Inference Device | NVIDIA Jetson Nano |
| Target FPS | Real-time |
| Plate Detection Accuracy | High (threshold-filtered) |

---

## 🏛️ Developed At

**Institute for Plasma Research (IPR), Gandhinagar**
Computer Vision Internship — January 2025 to May 2025

---

## 👤 Author

**Aslan Pathan**
- 📧 aslan13604@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/your-profile)
- 💻 [GitHub](https://github.com/Aslankhan13)

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
