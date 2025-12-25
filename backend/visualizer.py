import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # --- DESIGN: BROADCAST / MILITARY (v3.1 Refined) ---
        
        self.colors = {
            "Black":      (0, 0, 0),         
            "White":      (235, 235, 235),   # Slightly off-white for better screen look
            "Amber":      (0, 165, 255),     
            "Yellow":     (0, 255, 255),     
            "Cyan":       (255, 255, 0),     
            "Grey":       (80, 80, 80),      
        }
        
        # Fonts - Optimized for Fullscreen (Thicker = less pixelated)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.s_header = 0.85
        self.s_label = 0.75 
        self.s_footer = 0.8  # Larger for readability
        self.thick = 2       # Thicker strokes prevent aliasing artifacts

    def draw_tracker_overlay(self, canvas, detections, fps):
        h_cam, w_cam, _ = canvas.shape # 1080p
        
        # 1. Layout Structure
        header_h = 70
        footer_h = 70
        
        # Bars
        cv2.rectangle(canvas, (0, 0), (w_cam, header_h), self.colors["Black"], -1)
        cv2.rectangle(canvas, (0, h_cam - footer_h), (w_cam, h_cam), self.colors["Black"], -1)
        
        # --- HEADER (Top) ---
        y_head = 45
        
        # Icon + CAM-01 (Left)
        cv2.circle(canvas, (40, y_head - 12), 6, self.colors["Amber"], -1) 
        cv2.putText(canvas, "CAM-01", (60, y_head), 
                    self.font, self.s_header, self.colors["Amber"], self.thick, cv2.LINE_AA)
        
        # Center Title
        title = "OCEAN SENTRY"
        (tw, _), _ = cv2.getTextSize(title, self.font, self.s_header, self.thick)
        cv2.putText(canvas, title, (w_cam // 2 - tw // 2, y_head),
                    self.font, self.s_header, self.colors["White"], self.thick, cv2.LINE_AA)
                    
        # Right Stats
        fps_txt = f"REC [ {fps:.0f} ]" # Pseudo-REC style
        (fw, _), _ = cv2.getTextSize(fps_txt, self.font, self.s_header, self.thick)
        cv2.putText(canvas, fps_txt, (w_cam - fw - 40, y_head),
                    self.font, self.s_header, self.colors["White"], self.thick, cv2.LINE_AA)

        # --- CENTER CAMERA BORDER ---
        # 1280x960 centered
        # Issue fix: "Missing yellow side line" -> Make it 2px to be robust
        cam_w, cam_h = 1280, 960
        x_off = (w_cam - cam_w) // 2
        y_off = (h_cam - cam_h) // 2
        
        # Explicit coordinates
        pt1 = (x_off, y_off)
        pt2 = (x_off + cam_w, y_off + cam_h)
        cv2.rectangle(canvas, pt1, pt2, self.colors["Amber"], 2) # 2px thickness

        # --- DETECTIONS ---
        for det in detections:
            self.draw_single_detection(canvas, det)

        # --- FOOTER (Bottom Centered) ---
        # "LIVE ● | AI ACTIVE | OBJECTS: 3"
        
        # 1. Define Segments
        # Format: (Text, Color, IsDot?)
        
        ai_txt = "AI ACTIVE" if len(detections) > 0 else "AI IDLE"
        
        segments = [
            ("LIVE", self.colors["White"], True),      # Text + Red Dot
            (" | ", self.colors["Grey"], False),       # Separator
            (ai_txt, self.colors["White"], False),
            (" | ", self.colors["Grey"], False),
            (f"OBJECTS: {len(detections)}", self.colors["White"], False)
        ]
        
        # 2. Calculate Total Width
        total_width = 0
        seg_widths = []
        
        gap = 5 # spacing between text and dot
        
        for text, color, has_dot in segments:
            (w, h), _ = cv2.getTextSize(text, self.font, self.s_footer, self.thick)
            if has_dot:
                w += 20 # Add space for dot
            seg_widths.append(w)
            total_width += w
            
        # 3. Draw Centered
        cursor_x = (w_cam - total_width) // 2
        y_foot = h_cam - 25 # Baseline
        
        for i, (text, color, has_dot) in enumerate(segments):
            width = seg_widths[i]
            
            # Special case for LIVE dot
            if has_dot and text == "LIVE":
                cv2.putText(canvas, text, (cursor_x, y_foot), 
                            self.font, self.s_footer, color, self.thick, cv2.LINE_AA)
                
                # Dot position relative to text
                (tw, _), _ = cv2.getTextSize(text, self.font, self.s_footer, self.thick)
                dot_x = cursor_x + tw + 12
                cv2.circle(canvas, (dot_x, y_foot - 10), 6, (0, 0, 255), -1) # Red Rec Dot
                
            else:
                cv2.putText(canvas, text, (cursor_x, y_foot), 
                            self.font, self.s_footer, color, self.thick, cv2.LINE_AA)
            
            cursor_x += width # Advance cursor

        return canvas

    def draw_single_detection(self, canvas, det):
        l, t, r, b = det['box']
        label = det['label'].upper()
        score = det['score']
        
        color = self.colors["Yellow"]
        if label == "PLASTIC": color = self.colors["Cyan"]
        
        # Scout override
        if score < 0.45: color = self.colors["Grey"]
        
        # Box (Thicker for HD visibility)
        cv2.rectangle(canvas, (l, t), (r, b), color, 2)
        
        # Label (Solid)
        label_txt = f"{label} {score:.0%}"
        (tw, th), base = cv2.getTextSize(label_txt, self.font, self.s_label, self.thick)
        
        bg_l, bg_t = l, t - th - 12
        bg_r, bg_b = l + tw + 12, t
        
        if bg_t < 0:
            bg_t, bg_b = t, t + th + 12 # Flip down if offscreen
            
        cv2.rectangle(canvas, (bg_l, bg_t), (bg_r, bg_b), color, -1)
        
        # Text (Black high contrast)
        cv2.putText(canvas, label_txt, (bg_l + 6, bg_b - 6), 
                    self.font, self.s_label, self.colors["Black"], self.thick, cv2.LINE_AA)

    def draw_offline_screen(self, canvas):
        canvas.fill(0)
        h, w, _ = canvas.shape
        # Grid pattern for offline
        step = 50
        cv2.line(canvas, (0,h//2), (w,h//2), (20,20,20), 1)
        cv2.line(canvas, (w//2,0), (w//2,h), (20,20,20), 1)
        
        text = "NO SIGNAL"
        (tw, th), _ = cv2.getTextSize(text, self.font, 1.5, 3)
        cv2.putText(canvas, text, (w//2 - tw//2, h//2 + th//2), 
                    self.font, 1.5, (100,100,100), 3, cv2.LINE_AA)
