import cv2
import numpy as np
import time

class Visualizer:
    def __init__(self):
        # Professional Color Palette (BGR format for OpenCV)
        # Plastic: Neon Blue
        # Metal: Industrial Orange
        # Glass: Cyan / Ice
        # Trash: Warning Red
        self.colors = {
            "Plastic": (255, 191, 0),   # Deep Sky Blue
            "Metal": (0, 140, 255),     # Orange
            "Glass": (255, 255, 0),     # Cyan
            "Trash": (0, 0, 255),       # Red
            "Scout": (120, 120, 120),   # Gray (for low confidence)
            "Unknown": (128, 128, 128)  # Gray
        }
        
        # Font settings for high-tech look
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 1
        
        # Dashboard UI
        self.header_height = 60

    def draw_tracker_overlay(self, frame, detections, fps):
        """Draws a professional HUD overlay (Classic Pro Version)"""
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # 1. Darken image slightly for "Cinema Mode" feel (Optional, disabled for now)
        # overlay = cv2.addWeighted(overlay, 0.9, np.zeros(overlay.shape, overlay.dtype), 0, 0)

        # 2. Draw Top Dashboard (Semi-transparent black bar)
        cv2.rectangle(overlay, (0, 0), (w, self.header_height), (20, 20, 20), -1)
        
        # Add Header Text
        cv2.putText(overlay, "OCEAN SENTRY AI", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(overlay, "LIVE INFERENCE", (20, 52), self.font, 0.4, (0, 255, 0), 1)
        
        # Add FPS Counter (Right side)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 120, 35), self.font, 0.7, (200, 200, 200), 1)
        
        # Blending the dashboard
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 3. Draw Detections
        for det in detections:
            frame = self.draw_single_detection(frame, det)

        # 4. Status Line (Bottom) if nothing detected
        if not detections:
            cv2.putText(frame, "SCANNING SECTOR...", (w//2 - 80, h - 30), self.font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            # Show count
            stats = f"DETECTED: {len(detections)} OBJECTS"
            cv2.putText(frame, stats, (w//2 - 80, h - 30), self.font, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

        return frame

    def draw_single_detection(self, frame, det):
        left, top, right, bottom = det['box']
        label = det['label']
        score = det['score']
        
        # --- SCOUT MODE ---
        is_scout = False
        if score < 0.45:
            is_scout = True
            color = self.colors["Scout"]
            label_text = f"? ({score:.0%})" # Just a question mark
        else:
            color = self.colors.get(label, self.colors["Unknown"])
            label_text = f"{label} {score:.0%}"

        # 1. Draw Transparent Fill
        box_overlay = frame.copy()
        cv2.rectangle(box_overlay, (left, top), (right, bottom), color, -1)
        
        alpha_box = 0.1 if is_scout else 0.2  # Less visible if unsure
        frame = cv2.addWeighted(box_overlay, alpha_box, frame, 1 - alpha_box, 0)
        
        # 2. Draw Solid Border
        # Dashed line for scout? OpenCV doesn't support dashed rect easily.
        # We just make it thinner.
        thickness = 1 if is_scout else 2
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
        
        # 3. Draw Label Pill (Top Left)
        # For Scout, we keep it minimal
        (text_w, text_h), baseline = cv2.getTextSize(label_text, self.font, self.font_scale, self.thickness)
        
        # Draw Label Background
        cv2.rectangle(frame, (left, top - text_h - 10), (left + text_w + 10, top), color, -1)
        
        # Draw Text
        text_color = (255, 255, 255)
        if label == "Glass" and not is_scout: 
            text_color = (0, 0, 0)
            
        cv2.putText(frame, label_text, (left + 5, top - 5), self.font, self.font_scale, text_color, self.thickness, cv2.LINE_AA)
        
        return frame
