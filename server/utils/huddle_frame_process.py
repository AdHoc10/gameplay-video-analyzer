from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .yolo_detection import ImageDetection


class HuddleFrameProcessor:

    def __init__(self):
        self.min_players_for_huddle: int = 9
        # Setting luminance reference as the left and right regions of the central band (no overlay on them)
        self.luminance_reference: Optional[float] = None
        return

    # Compute and display multiple brightness metrics on the provided frames.
    def compute_metrics(
        self,
        frame: np.ndarray,
        prev_frame_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:

        # def roi(img, rx1, rx2, ry1, ry2):
        #     h, w = img.shape[:2]
        #     return img[int(ry1*h):int(ry2*h), int(rx1*w):int(rx2*w)]

        def roi(img, rx1, rx2, ry1, ry2):
            h, w = img.shape[:2]
            roi = img[int(ry1*h):int(ry2*h), int(rx1*w):int(rx2*w)]
            ref_reg_1 = img[int(ry1*h):int(ry2*h), 0:int(rx1*w)]
            ref_reg_2 = img[int(ry1*h):int(ry2*h), int(rx2*w):]
            reference_region = np.hstack((ref_reg_1, ref_reg_2))
            return roi, reference_region
        
        def overlay_score(bgr: np.ndarray) -> float:
            # Global darkness via luma (Y channel in YCrCb)
            ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            Y = ycc[:,:,0].astype(np.float32)/255.0
            dark = 1.0 - float(Y.mean())
            score = np.clip(dark, 0.0, 1.0)
            return float(score)

        bgr = frame

        # Panel region edge density (where translucent grid sits)
        panel, reference_region = roi(bgr, 0.12, 0.88, 0.35, 0.80)
        
        # Y (luma, Rec.601/709 style via OpenCV Y channel in YCrCb)
        Y = cv2.cvtColor(panel, cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)
        mean_Y = float(Y.mean())
        # median_Y = float(np.median(Y))
        # pct_black = float((Y < 20).mean()) * 100.0  # % pixels with Y<20
        
        # # HSV Value channel
        # V = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)[:,:,2].astype(np.float32)
        # mean_V = float(V.mean())
        
        # # HLS Lightness channel (OpenCV uses HLS ordering)
        # L = cv2.cvtColor(panel, cv2.COLOR_BGR2HLS)[:,:,1].astype(np.float32)
        # mean_L = float(L.mean())
        
        # # LAB L* channel
        # Lstar = cv2.cvtColor(panel, cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)  # 0..255 in OpenCV
        # mean_Lstar = float(Lstar.mean())
        
        # Overlay score (combined heuristic)
        o_score = overlay_score(panel) # (0, 1)
        
        cur_metrics = {
            "mean_Y": mean_Y,
            # "median_Y": median_Y,
            # "%Y<20": pct_black,
            # "mean_V(HSV)": mean_V,
            # "mean_L(HLS)": mean_L,
            # "mean_L*(LAB,0-255)": mean_Lstar,
            "overlay_score": o_score,
        }
        
        delta_metrics = {}
        if prev_frame_metrics:    
            for key in cur_metrics.keys():
                delta_metrics[key] = cur_metrics[key] - prev_frame_metrics[key] if prev_frame_metrics else float('inf')
        return cur_metrics, delta_metrics
    
    # Find a usable huddle frame from a cv2 capture variable (cv2 video seeker) 
    def find_proper_huddle_frame(
        self,
        cap: cv2.VideoCapture,
        huddle_frame_num: int,
        detector: ImageDetection,
    ) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, huddle_frame_num)
        num_people = 0
        while num_people < self.min_players_for_huddle:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detector.frame_detection(frame)
            num_people = len(boxes)
        prev_frame_metrics, metrics_delta = self.compute_metrics(frame)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_count += 1
            if frame_count % 3 == 0:
                frame_count = 0
                continue
            prev_frame_metrics, metrics_delta = self.compute_metrics(frame, prev_frame_metrics)
            if abs(metrics_delta["overlay_score"]) < 0.003 \
                or prev_frame_metrics["mean_Y"] > 90:
                # and abs(self.luminance_reference - prev_frame_metrics["mean_Y"]) < 20:
                break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
