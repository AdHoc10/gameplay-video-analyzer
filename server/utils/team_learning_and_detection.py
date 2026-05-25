from ultralytics import YOLO
from sklearn.mixture import GaussianMixture
import cv2
import numpy as np

from utils.yolo_detection import ImageDetection

import torch
from torchvision.models.resnet import resnet50, ResNet50_Weights # 18, 34 other options
# from torchvision.models import densenet121, DenseNet121_Weights


class TeamRepresentationLearning:

    def __init__(self, huddle_frames, yolo_model_path):
        self.yolo_detector = ImageDetection(yolo_model_path)
        self.huddle_frames = huddle_frames

        # Percents of the frame to shrink to focus more on the jerseys
        # Height: Central third, Same for width
        self.frac_top = 0.3
        self.frac_bot = 0.3
        self.frac_left = 0.1
        self.frac_right = 0.1

        # Load pre-trained model but exclude the fully connected layer 
        # Make sure to check the nomenclature of the final layers of the model and remove accordingly
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device used by PyTorch: {self.device}')

        # Load model onto device of choice (CPU or GPU)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity() # Setting the fully connected (fc) layer to identity
        # densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        # densenet.classifier = torch.nn.Identity()
        self.model = resnet
        self.model.to(self.device)

        print('Learning the team representations from the huddle frames...')
        self.team_gmm = self.train_gmm()
        print('Created representations of teams')


    def cnn_feature_extraction(self, img):
        torch.cuda.empty_cache()        
        # Prepare the image for inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(np.moveaxis(img, -1, 0), 0)
        img = torch.Tensor(img).to(self.device)
        # Extract features
        with torch.no_grad():
            feats = self.model(img)
        return feats.cpu()
    

    def compile_player_features_from_frame(self, frame):
        boxes = self.yolo_detector.frame_detection(frame)
        # boxes = self.yolo_detector.bboxes
        
        feats = []
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            if y2 < 250:
                continue
            w = x2 - x1
            h = y2 - y1
            x, y = x1, y1
            crop = frame[y:y+h, x:x+w]
            feat = self.cnn_feature_extraction(crop)
            feats.append(feat)
        return feats


    def compile_player_features_from_all_frames(self):
        huddle_feats = []
        for _, frame in self.huddle_frames.items():
            feats = self.compile_player_features_from_frame(frame)
            if feats:
                huddle_feats.extend(feats)
        huddle_feats = np.asarray(huddle_feats, dtype=np.float32).squeeze()
        return huddle_feats
    
    def train_gmm(self):

        huddle_feats = self.compile_player_features_from_all_frames()

        # ---------- fit a 2-component GMM ----------
        teams_gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",   # each team gets its own full covariance
            init_params="k-means++",
            reg_covar=1e-5,           # numerical stability
            n_init=5,
            max_iter=1000,
            random_state=42,
            # verbose=True
        )
        teams_gmm.fit(huddle_feats)

        # Optional: persist for later use
        # import joblib
        # joblib.dump(gmm, "team_gmm.joblib")
        return teams_gmm
    
    
    def classify_team(self, player_crop, probabilities=False):
        feat = self.cnn_feature_extraction(player_crop)
        if feat is None:
            # handle empty crop if needed
            return None
        
        # If you want a confidence score:
        if probabilities:    
            proba = self.teams_gmm.predict_proba(feat.reshape(1, -1))[0]  # length 2, sums to 1
            # confidence = float(np.max(proba))  # e.g., 0.0–1.0
            team_label = np.argmax(proba)
            return [team_label, proba]

        # Hard assignment (team 0 or 1)
        team_label = int(self.team_gmm.predict(feat.reshape(1, -1))[0])
        return [team_label]


    def define_defence(self, frame):
        
        # Tightly crop the frame to include players only
        # Find upper and lower bounds using player detections
        boxes = self.yolo_detector.frame_detection(frame)
        
        min_y = float('inf')
        max_y = 0
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            # To handle an anomalous detection of a team logo as person
            if y2<250:
                continue
            min_y = min(min_y, y1)
            max_y = max(max_y, y2)
            
        frame_crop = frame[min_y:max_y, :]
        
        # Taking bottom half of the tight crop to detect attacking team
        bottom_half_of_crop = frame_crop[len(frame_crop)//2:, :]
        crop_boxes = self.yolo_detector.frame_detection(bottom_half_of_crop)
        labels = []
        for box in crop_boxes:
            x1,y1,x2,y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            x, y = x1, y1

            player_crop = bottom_half_of_crop[y:y+h, x:x+w]
            pred = self.classify_team(player_crop)
            
            if pred is None:
                continue
            team_label = pred[0]
            probs = pred[1] if len(pred)==2 else None
            labels.append(team_label)
        
        vals, counts = np.unique(labels, return_counts=True)
        attacking_team = vals[np.argmax(counts)]
        return int(1 - attacking_team)
    
    
    def count_defenders(self, frame, defending_team):
        boxes = self.yolo_detector.frame_detection(frame)
        labels = []
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            x, y = x1, y1
            
            player_crop = frame[y:y+h, x: x+w]
            pred = self.classify_team(player_crop)
            if pred is None:
                continue
            team_label = pred[0]
            probs = pred[1] if len(pred)==2 else None
            labels.append(team_label)
        
        vals, counts = np.unique(labels, return_counts=True)
        # Counts[index of defending team]
        # Model is not always detecting defenders in the frame
        try:
            defenders_count = counts[vals.tolist().index(defending_team)]
        except:
            return 0
        
        return defenders_count