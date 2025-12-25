import numpy as np

class DetectionSmoother:
    def __init__(self, history_size=5, min_hits=2, smoothing_factor=0.3):
        self.history = [] # List of detection lists
        self.history_size = history_size
        self.min_hits = min_hits
        self.alpha = smoothing_factor # Lower = smoother but more lag
        
    def update(self, current_detections):
        self.history.append(current_detections)
        if len(self.history) > self.history_size:
            self.history.pop(0)
            
        if len(self.history) < 2:
            return current_detections # Not enough history yet
            
        # Box Smoothing (Exponential Moving Average)
        # We perform simple IoU matching to find the "same" box in the previous frame.
        
        smoothed_detections = []
        prev_detections = self.history[-2]
        
        for curr in current_detections:
            # Find best match in previous frame
            best_match = None
            max_iou = 0.0
            
            curr_box = curr['box']
            cx1, cy1, cx2, cy2 = curr_box
            curr_area = (cx2 - cx1) * (cy2 - cy1)
            
            for prev in prev_detections:
                px1, py1, px2, py2 = prev['box']
                
                # Calculate IoU
                xx1 = max(cx1, px1)
                yy1 = max(cy1, py1)
                xx2 = min(cx2, px2)
                yy2 = min(cy2, py2)
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h
                
                prev_area = (px2 - px1) * (py2 - py1)
                union_area = curr_area + prev_area - inter_area
                
                if union_area > 0:
                    iou = inter_area / union_area
                    # Threshold: if overlap is > 30%, assume it's the same object
                    if iou > 0.3 and iou > max_iou:
                        # Also check if label matches!
                        if prev['label'] == curr['label']:
                            max_iou = iou
                            best_match = prev
            
            if best_match:
                # Smooth the box
                px1, py1, px2, py2 = best_match['box']
                
                nx1 = int(px1 * (1-self.alpha) + cx1 * self.alpha)
                ny1 = int(py1 * (1-self.alpha) + cy1 * self.alpha)
                nx2 = int(px2 * (1-self.alpha) + cx2 * self.alpha)
                ny2 = int(py2 * (1-self.alpha) + cy2 * self.alpha)
                
                curr['box'] = (nx1, ny1, nx2, ny2)
                
            smoothed_detections.append(curr)
            
        return smoothed_detections
