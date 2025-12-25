import cv2
import numpy as np
try:
    # TRY LITE RUNTIME FIRST (Better for Pi)
    from tflite_runtime.interpreter import Interpreter
    print("DEBUG: Using tflite_runtime (Lightweight) ✅")
except ImportError:
    # Fallback to full TensorFlow (slower, heavier)
    print("DEBUG: tflite_runtime not found, falling back to full TensorFlow... ⚠️")
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

import os

class TrashDetector:
    def __init__(self, model_path="models/model.tflite", label_path="models/labels.txt"):
        # Enable 4 threads for Quad-Core Pi
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        # Label Mapping (Simplify names)
        self.label_map = {
            "Clear plastic bottle": "Plastic Bottle",
            "Single-use carrier bag": "Plastic Bag",
            "Drink can": "Metal Can",
            "Food Can": "Metal Can",
            "Aerosol": "Metal Can",
            "Disposable plastic cup": "Plastic Cup",
            "Plastic bottle cap": "Cap",
            "Metal bottle cap": "Cap",
            "Pop tab": "Metal Can", 
        }
        
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        print(f"DEBUG: Loaded {len(self.labels)} labels. First 5: {self.labels[:5]}")

    def detect(self, frame, threshold=0.5, enhance=True):
        """
        enhance=True: Activates 'Crystal Vision' (CLAHE) for better detection in murky water.
        """
        if enhance:
            frame = self.apply_enhancement(frame)

        frame_resized = cv2.resize(frame, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        # Normalize input (YOLOv8 expects 0-1 float32)
        if self.input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        else:
            input_data = np.uint8(input_data)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # --- FACE FILTER (Identity Shield) ---
        if not hasattr(self, 'face_cascade'):
             cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
             self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        # -------------------------------------

        detections = []
        
        # Check if it's YOLOv8 (1 output) or SSD (4 outputs)
        if len(self.output_details) == 1:
            # YOLOv8 Output: [1, 4 + num_classes, 8400]
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            # Transpose to [8400, 4 + num_classes]
            output = output.transpose()
            
            boxes = []
            confidences = []
            class_ids = []
            
            # Iterate through all 8400 candidates
            # Optimization: Vectorized filtering
            scores = np.max(output[:, 4:], axis=1)
            
            # DEBUG: Print max score to see if model is detecting anything
            if len(scores) > 0:
                max_score = np.max(scores)
                if max_score > 0.1: # Only print if somewhat confident
                    print(f"DEBUG: Max Score: {max_score:.2f}")

            mask = scores > threshold
            
            filtered_output = output[mask]
            filtered_scores = scores[mask]
            
            for i, row in enumerate(filtered_output):
                # Box: cx, cy, w, h
                cx, cy, w, h = row[:4]
                
                # Convert to top-left x, y
                left = int((cx - w/2) * frame.shape[1])
                top = int((cy - h/2) * frame.shape[0])
                width = int(w * frame.shape[1])
                height = int(h * frame.shape[0])
                
                # Get class
                classes_scores = row[4:]
                class_id = np.argmax(classes_scores)
                
                boxes.append([left, top, width, height])
                confidences.append(float(filtered_scores[i]))
                class_ids.append(class_id)
                
            # Apply NMS (Non-Maximum Suppression)
            # Threshold 0.65 allows more overlap (e.g. bottle behind bottle)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.65)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    left, top, width, height = box[0], box[1], box[2], box[3]
                    label = self.labels[class_ids[i]] if class_ids[i] < len(self.labels) else "Unknown"
                    
                    if label in self.label_map:
                        label = self.label_map[label]
                    
                    det_box = (left, top, left + width, top + height)
                    
                    # --- FACE FILTER CHECK ---
                    is_face = False
                    for (fx, fy, fw, fh) in faces:
                        f_box = (fx, fy, fx+fw, fy+fh)
                        xA = max(det_box[0], f_box[0])
                        yA = max(det_box[1], f_box[1])
                        xB = min(det_box[2], f_box[2])
                        yB = min(det_box[3], f_box[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        objArea = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                        if interArea > 0.3 * objArea: # 30% overlap with face -> ignore
                           is_face = True
                           break
                    # -------------------------
                    
                    if not is_face:
                        detections.append({
                            "box": det_box,
                            "label": label,
                            "score": confidences[i]
                        })

        else:
            # Legacy SSD MobileNet Logic
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

            for i in range(len(scores)):
                if scores[i] > threshold:
                    class_id = int(classes[i])
                    label = self.labels[class_id] if class_id < len(self.labels) else "Unknown"
                    
                    if whitelist and label not in whitelist:
                        continue

                    ymin, xmin, ymax, xmax = boxes[i]
                    im_h, im_w, _ = frame.shape
                    (left, right, top, bottom) = (xmin * im_w, xmax * im_w, ymin * im_h, ymax * im_h)
                    
                    detections.append({
                        "box": (int(left), int(top), int(right), int(bottom)),
                        "label": label,
                        "score": float(scores[i])
                    })
        
        return detections

    def detect_tiled(self, frame, threshold=0.5):
        """Runs detection on 4 quadrants of the frame for higher accuracy"""
        height, width, _ = frame.shape
        half_h, half_w = height // 2, width // 2
        
        # Define quadrants
        quadrants = [
            (0, 0, half_w, half_h),          # Top-Left
            (half_w, 0, width, half_h),      # Top-Right
            (0, half_h, half_w, height),     # Bottom-Left
            (half_w, half_h, width, height)  # Bottom-Right
        ]
        
        all_detections = []
        
        for x_off, y_off, x_end, y_end in quadrants:
            crop = frame[y_off:y_end, x_off:x_end]
            results = self.detect(crop, threshold)
            
            # Adjust coordinates
            for res in results:
                box = res['box']
                new_box = (box[0] + x_off, box[1] + y_off, box[2] + x_off, box[3] + y_off)
                res['box'] = new_box
                all_detections.append(res)
                
        return all_detections

    def draw_detections(self, frame, detections):
        for det in detections:
            left, top, right, bottom = det['box']
            label = f"{det['label']} {det['score']:.2f}"
            
            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            
            # Draw label background
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line), (0, 255, 255), cv2.FILLED)
            
            # Draw text
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame

    def apply_enhancement(self, frame):
        """
        'Crystal Vision': Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to specific channels to enhance contrast in murky/brown water without distorting colors too much.
        """
        # Convert to LAB color space (L = Lightness, A/B = Colors)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel (Lightness) only
        # clipLimit=2.0 -> Moderate contrast enhancement
        # tileGridSize=(8,8) -> Local area size
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

class EnsembleDetector:
    """Combines detections from multiple models"""
    def __init__(self, detectors):
        self.detectors = detectors
        
    def detect(self, frame, threshold=0.5):
        all_detections = []
        
        # Run all models
        for detector in self.detectors:
            dets = detector.detect(frame, threshold)
            all_detections.extend(dets)
            
        # Merge duplicates using NMS
        if not all_detections:
            return []
            
        boxes = [d['box'] for d in all_detections]
        scores = [d['score'] for d in all_detections]
        
        # We need to handle labels. For NMS, we usually do it per class, 
        # but here we want to merge "Can" from Model A and "Can" from Model B.
        # Simplified NMS: Just merge everything based on box overlap.
        
        # Convert to format for cv2.dnn.NMSBoxes: (x, y, w, h)
        nms_boxes = []
        for (x1, y1, x2, y2) in boxes:
            nms_boxes.append([x1, y1, x2-x1, y2-y1])
            
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, threshold, 0.45)
        
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append(all_detections[i])
                
        return final_detections

    def detect_tiled(self, frame, threshold=0.5):
        # For ensemble, tiling might be too slow (2 models * 4 tiles = 8x inference).
        # Let's just run standard detect for now.
        return self.detect(frame, threshold)
