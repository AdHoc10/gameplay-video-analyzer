from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class ImageDetection:

    def __init__(self, model_path: str):
        self.yolo_model: YOLO = YOLO(model_path)
        self.bboxes: Optional[List[np.ndarray]] = None
        self.img_height: Optional[int] = None
        self.img_width:  Optional[int] = None
        self.category_id: int = 0  # COCO class 0 = person

    # Function that returns bboxes of person/people in a frame
    def frame_detection(self, img: np.ndarray, CONF_TH: float = 0.3) -> List[np.ndarray]:
        '''
            Convert img to suitable color code before passing
        '''
        res = self.yolo_model.predict(img, classes=[self.category_id], verbose=False)[0]  # class 0 = person
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        confident_boxes = []
        for b, c in zip(boxes, confs):
            if c >= CONF_TH:
                # confident_boxes.append(b.tolist())  # [x1,y1,x2,y2]
                confident_boxes.append(b)
        
        # self.bboxes = confident_boxes
        
        try:
            self.img_height = img.shape[0]
            self.img_width = img.shape[1]
        except:
            self.img_height = len(img)
            self.img_width = len(img[0])
        return confident_boxes
    
    
    def detect_and_save_annotations(self, frame, save_dir: str, img_name: str) -> None:  # Annotation file should have the same name as the img file
        if type(frame) == 'str':
            frame = cv2.imread(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.frame_detection(frame)
        
        name = img_name.split('.')[0]
        annot_name = f'{name}.txt'
        annot_filepath = f'{save_dir}/{annot_name}'
        with open(annot_filepath, 'w') as f:
            for box in self.bboxes:
                x1,y1,x2,y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                x, y = x1, y1
                x_center = (x + w/2) / self.img_width 
                y_center = (y + h/2) / self.img_height
                width = w / self.img_width
                height = h / self.img_height
                f.write(f"{self.category_id} {x_center} {y_center} {width} {height}\n")
        return
    



    
    
    
