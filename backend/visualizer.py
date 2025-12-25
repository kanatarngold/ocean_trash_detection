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
        
        # Font settings for high-tech look (Anti-aliased)
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        self.font_scale = 0.6
        self.thickness = 1
        
        # Dashboard UI
        self.header_height = 60

    def draw_tracker_overlay(self, frame, detections, fps):
        """Draws a professional HUD overlay (HD Ready)"""
        h, w, _ = frame.shape
        
        # Dynamic Scaling for HD (1280px) vs SD (640px)
        scale = 1.0
        if w > 1000:
            scale = 2.0
            
        header_h = int(60 * scale)
        font_sc = self.font_scale * scale * 0.9
        thick = max(1, int(self.thickness * scale))

        # 1. Solid Black Header (Requested by User)
        cv2.rectangle(frame, (0, 0), (w, header_h), (0, 0, 0), -1)
        
        # Header Text
        cv2.putText(frame, "OCEAN SENTRY AI", (int(20*scale), int(35*scale)), cv2.FONT_HERSHEY_DUPLEX, font_sc * 1.2, (255, 255, 255), thick)
        cv2.putText(frame, "LIVE SURVEILLANCE", (int(20*scale), int(52*scale)), self.font, font_sc * 0.7, (0, 255, 0), thick)
        
        # Fullscreen Hint
        cv2.putText(frame, "[F] Fullscreen", (int(350*scale), int(35*scale)), self.font, font_sc * 0.8, (100, 100, 100), thick)
        
        # FPS Counter (Right side)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - int(150*scale), int(35*scale)), self.font, font_sc, (200, 200, 200), thick)
        
        # 2. Draw Detections
        for det in detections:
            frame = self.draw_single_detection(frame, det, scale)

        # 3. Solid Black Footer (Status Line)
        cv2.rectangle(frame, (0, h - int(40*scale)), (w, h), (0, 0, 0), -1)
        
        if not detections:
            cv2.putText(frame, "SCANNING SECTOR...", (w//2 - int(100*scale), h - int(12*scale)), self.font, font_sc, (0, 255, 0), thick, cv2.LINE_AA)
        else:
            stats = f"DETECTED: {len(detections)} OBJECTS"
            cv2.putText(frame, stats, (w//2 - int(100*scale), h - int(12*scale)), self.font, font_sc, (0, 200, 255), thick, cv2.LINE_AA)

        return frame

    def draw_single_detection(self, frame, det, scale=1.0):
        left, top, right, bottom = det['box']
        label = det['label']
        score = det['score']
        
        # Scaling
        font_sc = self.font_scale * scale
        thick = max(1, int(self.thickness * scale))
        
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
        box_thick = 1 if is_scout else thick
        cv2.rectangle(frame, (left, top), (right, bottom), color, box_thick)
        
        # 3. Draw Label Pill (Top Left)
        (text_w, text_h), baseline = cv2.getTextSize(label_text, self.font, font_sc, thick)
        
        # Draw Label Background
        pill_pad = int(10 * scale)
        cv2.rectangle(frame, (left, top - text_h - pill_pad), (left + text_w + pill_pad, top), color, -1)
        
        # Draw Text
        text_color = (255, 255, 255)
        if label == "Glass" and not is_scout: 
            text_color = (0, 0, 0)
            
        cv2.putText(frame, label_text, (left + int(5*scale), top - int(5*scale)), self.font, font_sc, text_color, thick, cv2.LINE_AA)
        
        return frame
