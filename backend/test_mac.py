import cv2
import time
import sys
import os

# Füge aktuellen Ordner zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from detector import TrashDetector
    from visualizer import Visualizer 
    from tracker import DetectionSmoother # RE-ENABLED
except ImportError:
    print("Fehler: konnte Module (detector, visualizer, tracker) nicht finden.")
    sys.exit(1)

# Konfiguration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_int8.tflite")
LABEL_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

def main():
    print("🖥️ Starte PRO-Test auf Mac (Synchron + Overlap Fix)...")

    # 1. Detector, Visualizer, Tracker laden
    try:
        detector = TrashDetector(model_path=MODEL_PATH, label_path=LABEL_PATH)
        visualizer = Visualizer()
        smoother = DetectionSmoother(history_size=5, smoothing_factor=0.3)
        print("✅ Modell, UI & Tracker geladen!")
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return

    # 2. Webcam
    cap = cv2.VideoCapture(0)
    # Höhere Auflösung für Mac
    cap.set(3, 1280)
    cap.set(4, 720)

    print("🎥 Kamera läuft! Drücke 'q' zum Beenden.")
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference (Synchron - Wartet auf KI für perfekte Position)
        # Threshold 0.25 -> "Scout Mode" aktiv
        detections = detector.detect(frame, threshold=0.25)
        
        # Stabilisierung
        detections = smoother.update(detections)

        # FPS Calculation
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 1.0: 
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Draw Professional Overlay
        frame = visualizer.draw_tracker_overlay(frame, detections, fps)

        cv2.imshow('Mac Test - Ocean Trash Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
