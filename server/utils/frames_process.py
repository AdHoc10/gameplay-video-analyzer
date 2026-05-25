from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .csv_process import CSVVideoInfoProcessor
from .yolo_detection import ImageDetection
from .huddle_frame_process import HuddleFrameProcessor

from utils.team_learning_and_detection import TeamRepresentationLearning


class FrameHandler(CSVVideoInfoProcessor, HuddleFrameProcessor):

    def __init__(
        self,
        move_annot_filepath: str,
        huddle_annot_filepath: str,
        VIDEOS_DIR: str,
        yolo_model_path: Optional[str] = None,
    ):
        CSVVideoInfoProcessor.__init__(self, move_annot_filepath, huddle_annot_filepath, VIDEOS_DIR)
        if not self.video_exists_on_yt:
            print(f'Error downloading URL {self.video_url}')
            return
        HuddleFrameProcessor.__init__(self)

        # Map each huddle frame to the move frames in the corresponding play
        # Video fps required to scrub through the video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.huddle_to_moves_map = self.match_huddle_times_with_moves(fps)
        
        # Map the huddle frame number to the relevant (a decent huddle frame that can be used) frame 
        print()
        print('Loading and storing suitable huddle frames')
        self.huddle_frames = self.store_huddle_frames(yolo_model_path)

        return
  

    # Function to convert time to frame number
    def convert_timestamp_to_frames(self, timestamp: str, fps: float) -> int:
        ''' 
            Converts a timestamp to frame number (time in seconds * fps).
            Args:
                time: timestamp in format hh:mm:ss.ms, mm:ss.ms or ss.ms
                fps: frames per second of the video
            Returns:
                int: frame number
        '''
        # Splitting the timestamp to hours, minutes, seconds
        timestamp_elements = timestamp.split(':')
        # Reversing to make calculations easier
        timestamp_elements.reverse()
        
        # Converting the timestamp to seconds
        time_in_seconds = sum(float(x) * 60 ** i for i, x in enumerate(timestamp_elements))
        # time_in_seconds = max(0, time_in_seconds - 0.2)     # Subtracting 0.2 seconds from the start time
        
        # Returning the time in frames (time in seconds * fps)
        return max(int(time_in_seconds * fps - 3), 0)
    

    # Defining huddle_frame to move_frame dict: {huddle_frame_number: [(move_tag, move_frame_num, down_number), ...], ...}
    def match_huddle_times_with_moves(
        self,
        fps: float = 60,
    ) -> Dict[int, List[Tuple[str, int, Optional[int]]]]:
        ''' 
            Creates a dictionary that stores huddle times as dict keys and list of moves made in the corresponding play as dict values.
            Considers only ATTACKING MOVES for now 
            Uses:
                move_annotations: dataframe containing annotations of moves
                huddle_annotations: dataframe containing timestamps of start times
            Returns:
                dict: {start_time_frame_number: [(move_tagname, move_frame_number), ...], ... }
        '''
        huddle_times = self.huddle_annotations['Timestamps'].tolist()
        huddle_to_moves_map = {self.convert_timestamp_to_frames(start_time, fps): [] for start_time in huddle_times}
        
        for _, row in self.move_annotations.iterrows():
            move_start_time = row['StartTime']
            move_tag = row['TagName']
            down_number = row['Down']

            # Only checking for attacking moves FOR NOW
            if move_tag not in ["SPIN", "JUKE", "PASS_THROW", "WALL_MOVE", "STIFF_ARM", "HURDLE"]:
                continue

            # Convert move start time to frames
            move_start_frame = self.convert_timestamp_to_frames(move_start_time, fps)

            # Find the closest start frame that is less than or equal to the action start frame
            closest_huddle_time = None
            for huddle_time in huddle_to_moves_map.keys():
                if huddle_time <= move_start_frame:
                    if closest_huddle_time is None or huddle_time > closest_huddle_time:
                        closest_huddle_time = huddle_time

            if closest_huddle_time is not None:
                huddle_to_moves_map[closest_huddle_time].append((move_tag, move_start_frame, down_number))
                huddle_to_moves_map[closest_huddle_time] = sorted(huddle_to_moves_map[closest_huddle_time], key=lambda x: x[1])
        
        return huddle_to_moves_map

    
    # # Function to set relevant frame luminance reference (expected range)
    # def set_luminance_reference(self, move_frame_num=None):
    #     # Open video
    #     cap = cv2.VideoCapture(self.video_path)
    #     # fps = int(cap.get(cv2.CAP_PROP_FPS))
    #     # print(f"Video FPS: {fps}")
        
    #     if move_frame_num == None:
    #         try:
    #             _, move_frame_num = list(self.huddle_to_moves_map.values())[0][1]
    #         except Exception as e:
    #             print(list(self.huddle_to_moves_map.values()))
    #             raise e
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, move_frame_num)
    #     while not ret:
    #         ret, frame = cap.read()
    #     cur_metrics, _ = self.compute_metrics(frame)
    #     self.luminance_reference = cur_metrics["mean_Y"]
    #     cap.release()
    #     return
    

    # Function to load huddle frames (Optionally load move frames)
    # Dict keys: start frames, Dict values: list of (action_tag, action_frame) in that play
    def store_huddle_frames(self, model_path: Optional[str] = None) -> Dict[int, np.ndarray]:
        
        # Load required detection model
        detector = ImageDetection(model_path)

        # Open video
        cap = cv2.VideoCapture(self.video_path)

        huddle_frames = {}
        # last_move_frame_num_before_huddle = None
        # Start processing the video following the start_time to moves dictionary    
        for i, (huddle_frame_num, moves_list) in enumerate(tqdm(self.huddle_to_moves_map.items())):
            if len(moves_list)==0:
                continue
            
            # if not last_move_frame_num_before_huddle:
            #     last_move_frame_num_before_huddle = moves_list[0][1]
            # self.set_luminance_reference(last_move_frame_num_before_huddle)
            # last_move_frame_num_before_huddle = moves_list[-1][1]

            huddle_frames[huddle_frame_num] = self.find_proper_huddle_frame(cap, huddle_frame_num, detector)

        cap.release()
        del(detector)
        return huddle_frames
    

    # Loading start_frame to action_frame dict: {start_frame_number: [(action_tag, action_frame), ...], ...}
    # Dict keys: start frames, Dict values: list of (action_tag, action_frame) in that play
    def analyze_move_frames(self, team_representations: "TeamRepresentationLearning"):
            
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video FPS: {fps}")

        moves_2_defenderCount_dict = {}
        moves_2_timeSincePlayBegan = {}

        # Start processing the video following the start_time to moves dictionary    
        for i, (start_time_frame, moves) in enumerate(tqdm(self.huddle_to_moves_map.items())):
            if not moves:
                continue
            
            huddle_frame = self.huddle_frames[start_time_frame]
            defending_team = team_representations.define_defence(huddle_frame)
            
            for move in moves:
                # Each move is of the form (action_tag, action_frame_number, down_number)
                (action_tag, move_frame_number, down_number) = move

                cap.set(cv2.CAP_PROP_POS_FRAMES, move_frame_number)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Count defenders in frame
                defenders_count = team_representations.count_defenders(frame, defending_team)
                # Save move to defender count in a dictionary
                moves_2_defenderCount_dict[move_frame_number] = (action_tag, defenders_count)    
                # Time in seconds (whole numbers) 
                # moves_2_timeSincePlayBegan[move_frame_number] = (action_tag, round((move_frame_number - start_time_frame)/fps, 3), down_number)
                # Time in frames (multiples of 60)
                moves_2_timeSincePlayBegan[move_frame_number] = (action_tag, move_frame_number - start_time_frame, down_number)     
                
        cap.release() 
        return moves_2_defenderCount_dict, moves_2_timeSincePlayBegan
    
