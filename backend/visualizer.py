import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # --- DESIGN: BROADCAST / MILITARY (Reference Image) ---
        # High Contrast, True Black, Solid Labels
        
        # BGR Colors
        self.colors = {
            "Black":      (0, 0, 0),         # True Black
            "White":      (255, 255, 255),   # Pure White
            "Amber":      (0, 165, 255),     # #FFA500 (Orange/Amber Title)
            "Yellow":     (0, 255, 255),     # Standard Yellow
            "Cyan":       (255, 255, 0),     # Standard Cyan
            "Grey":       (80, 80, 80),      # Dark Grey for separators
        }
        
        # Fonts - Larger and bolder
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # Scales
        self.s_header = 0.8
        self.s_label = 0.7
        self.s_footer = 0.65

    def draw_tracker_overlay(self, canvas, detections, fps):
        """
        Layout based on reference:
        - Top Bar (Black) with "CAM-01", Icon, Title, FPS.
        - Center: Camera Feed with thin border.
        - Bottom Bar (Black) with detailed stats separated by pipes.
        """
        h_cam, w_cam, _ = canvas.shape # 1080p
        
        # 1. Background Fill (Ensure True Black borders)
        # Assuming inference_pi fills canvas with black or we overwrite headers
        # We need specific bars.
        
        header_h = 60
        footer_h = 60
        
        # Top Header Bar
        cv2.rectangle(canvas, (0, 0), (w_cam, header_h), self.colors["Black"], -1)
        # Bottom Footer Bar
        cv2.rectangle(canvas, (0, h_cam - footer_h), (w_cam, h_cam), self.colors["Black"], -1)
        
        # --- HEADER CONTENT ---
        y_head = 40
        # "CAM-01" (Orange/Amber) + Icon (Circle)
        cv2.circle(canvas, (30, y_head - 10), 8, self.colors["White"], -1) # "Icon" placeholder
        cv2.circle(canvas, (30, y_head - 10), 4, self.colors["Black"], -1) # abstract eye
        
        cv2.putText(canvas, "CAM-01", (60, y_head), 
                    self.font, self.s_header, self.colors["Amber"], 2, cv2.LINE_AA)
        
        # Center Title "Pallescreen" (or Project Name)
        title = "OCEAN SENTRY"
        (tw, _), _ = cv2.getTextSize(title, self.font, self.s_header, 2)
        cv2.putText(canvas, title, (w_cam // 2 - tw // 2, y_head),
                    self.font, self.s_header, self.colors["White"], 1, cv2.LINE_AA)
                    
        # Right "PPS: 8:0" (FPS)
        fps_txt = f"FPS: {fps:.1f}"
        (fw, _), _ = cv2.getTextSize(fps_txt, self.font, self.s_header, 2)
        cv2.putText(canvas, fps_txt, (w_cam - fw - 30, y_head),
                    self.font, self.s_header, self.colors["White"], 2, cv2.LINE_AA)

        # --- CAMERA BORDER ---
        # Image shows a thin yellow/orange border framing the view
        cam_w, cam_h = 1280, 960
        x_off = (w_cam - cam_w) // 2
        y_off = (h_cam - cam_h) // 2
        
        cv2.rectangle(canvas, (x_off, y_off), (x_off + cam_w, y_off + cam_h), 
                      self.colors["Amber"], 1)

        # --- DETECTIONS ---
        for det in detections:
            self.draw_single_detection(canvas, det)

        # --- FOOTER CONTENT ---
        # Style: "CAM-01 | LIVE ● | PAS 8 | 'AI ACTIVE | OBJECTS: 3"
        y_foot = h_cam - 20
        
        # Helper for piping
        cursor_x = 50
        
        def draw_segment(text, color=(255,255,255), bold=1):
            nonlocal cursor_x
            cv2.putText(canvas, text, (cursor_x, y_foot), 
                        self.font, self.s_footer, color, bold, cv2.LINE_AA)
            (w, _), _ = cv2.getTextSize(text, self.font, self.s_footer, bold)
            cursor_x += w + 20
            # Separator
            cv2.putText(canvas, "|", (cursor_x, y_foot), 
                        self.font, self.s_footer, self.colors["Grey"], 1, cv2.LINE_AA)
            cursor_x += 20

        draw_segment("CAM-01")
        
        # LIVE with Dot
        cv2.putText(canvas, "LIVE", (cursor_x, y_foot), self.font, self.s_footer, self.colors["White"], 1, cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize("LIVE", self.font, self.s_footer, 1)
        cursor_x += w + 10
        cv2.circle(canvas, (cursor_x, y_foot - 10), 6, self.colors["Amber"], -1) # Orange Dot
        cursor_x += 20
        cv2.putText(canvas, "|", (cursor_x, y_foot), self.font, self.s_footer, self.colors["Grey"], 1, cv2.LINE_AA)
        cursor_x += 20
        
        # AI Status
        ai_active = len(detections) > 0
        ai_txt = "'AI ACTIVE" if ai_active else "'AI IDLE"
        draw_segment(ai_txt)
        
        # Object Count
        draw_segment(f"OBJECTS: {len(detections)}")


    def draw_single_detection(self, canvas, det):
        """Solid Label Background (Black Text) + Thick Box"""
        l, t, r, b = det['box']
        label = det['label'].upper()
        score = det['score']
        
        # Colors based on class (simulating the yellow/cyan mix in image)
        # Plastic -> Cyan? Trash -> Yellow?
        if label == "PLASTIC":
            color = self.colors["Cyan"]
        elif label == "METAL":
            color = (0, 0, 255) # Red? Or sticking to ref image Palette
            color = self.colors["Yellow"] 
        else:
            color = self.colors["Yellow"]
            
        # Scout override
        if score < 0.45:
            color = self.colors["Grey"]
        
        # 1. Box
        cv2.rectangle(canvas, (l, t), (r, b), color, 2)
        
        # 2. Label (Solid Background)
        label_txt = f"{label} {score:.0%}"
        (tw, th), base = cv2.getTextSize(label_txt, self.font, self.s_label, 2)
        
        # Draw filled box ABOVE the rect
        # Ensure it fits
        bg_l = l
        bg_t = t - th - 10
        bg_r = l + tw + 10
        bg_b = t
        
        if bg_t < 0: # If box at top edge, draw label inside
            bg_t = t
            bg_b = t + th + 10
            
        cv2.rectangle(canvas, (bg_l, bg_t), (bg_r, bg_b), color, -1) # Filled
        
        # Text (Black)
        text_y = bg_b - 5
        cv2.putText(canvas, label_txt, (bg_l + 5, text_y), 
                    self.font, self.s_label, self.colors["Black"], 2, cv2.LINE_AA)

    def draw_offline_screen(self, canvas):
        canvas[:] = 0
        text = "NO SIGNAL"
        (tw, th), _ = cv2.getTextSize(text, self.font, 1.5, 2)
        h, w, _ = canvas.shape
        cv2.putText(canvas, text, (w//2 - tw//2, h//2), self.font, 1.5, (100,100,100), 2)
