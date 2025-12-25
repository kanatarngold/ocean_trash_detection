from ultralytics import YOLO

# 1. Standard-Modell laden (Nano - sehr schnell)
print("Lade Standard YOLOv8n...")
model = YOLO('yolov8n.pt') 

# 2. Exportieren zu TFLite (für Raspberry Pi)
print("Exportiere zu TFLite...")
model.export(format='tflite', int8=True)

print("\n✅ FERTIG!")
print("Deine Datei liegt hier: 'yolov8n_saved_model/yolov8n_int8.tflite'")
print("Benenne sie um in 'best_int8.tflite' und kopiere sie auf den Pi!")
