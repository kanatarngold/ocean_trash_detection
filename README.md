# 🌊 Ocean Trash Detector

Professional AI-powered system for detecting marine debris in real-time. Optimized for Raspberry Pi and Edge deployment.

## 🚀 Quick Start for Team Members

### 1. Installation
Ensure you have **Python 3.9+**.
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Locally (Mac/Windows)
Test the detection with your webcam using the stabilized Pro-Interface:

```bash
cd backend
python3 test_mac.py
```

### 3. Deploy to Raspberry Pi
Transfer the `backend` folder to your Pi and run:

```bash
cd backend
python3 inference_pi.py
```

## ✨ Features
*   **Sticky Boxes:** Stabilized tracking for smooth visualization.
*   **Identity Shield:** Automatically filters out human faces to prevent false positives.
*   **Crystal Vision (CLAHE):** Enhances contrast in murky/brown water.
*   **Scout Mode:** Displays low-confidence detections (25%+) as gray "?" boxes.
*   **Professional HUD:** Clean, high-performance overlay with FPS counter.

## 🛠️ Project Structure
*   `backend/`: Core logic, models, and inference scripts.
*   `backend/models/`: Contains `best_int8.tflite` (Quantized Model) and `labels.txt`.
*   `train_local.py`: Script for training a new custom model (YOLOv8).
