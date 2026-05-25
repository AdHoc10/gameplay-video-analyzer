import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from config import *
from utils.frames_process import FrameHandler
from utils.tracking import Tracking
from utils.tracking_refiner.refined_tracking import RefinedTracking
from utils.probability_creation import SaveEventDataAndProbabilities
from utils.geometry import iou  # re-exported for callers that imported it from here

from utils.tracking_refiner import RefinedTracking
from utils.observers import (
    BallCarrierObserver, BirdsEyeHeatmapObserver,
    BirdsEyeConfig, PerspectiveConfig,
)


def process_from_app(csv_name):

    def print_line_to_fill_terminal(character='-'):
        """
        Prints a single character repeatedly to fill the entire width of the terminal.

        Args:
            character (str): The character to be printed. Defaults to '-'.
        """
        try:
            # Get the terminal width
            terminal_width = shutil.get_terminal_size().columns
            
            # Print the character repeated to fill the width
            print(character * terminal_width)
        except OSError:
            # Handle cases where terminal size cannot be determined (e.g., non-interactive environments)
            print("Warning: Could not determine terminal size. Printing a default line.")
            print(character * 80) # Print a default length line
        return
    
    print(csv_name)

    # ----------------  data loader  -------------------------
    yolo_model_path = './models/yolo11m.pt'
    move_annot_filepath = f'{MOVE_ANNOT_FOLDER}/{csv_name}'
    huddle_annot_filepath = f'{START_TIMES_ANNOT_FOLDER}/start times {csv_name}'
    frame_handler = FrameHandler(move_annot_filepath, huddle_annot_filepath, VIDEOS_DIR, yolo_model_path)
    if not getattr(frame_handler, "video_exists_on_yt", False):
        raise RuntimeError(f"Failed to download source video: {getattr(frame_handler, 'video_url', 'unknown')}")
    if not hasattr(frame_handler, "huddle_to_moves_map"):
        raise RuntimeError("Frame handler initialization failed before huddle mapping.")

    # Start tracking algo
    tracker = RefinedTracking(csv_name, frame_handler)

    # --- CALIBRATION CADENCE TOGGLE (edit this one line) ---
    persp_cfg = PerspectiveConfig(mode="cached", cache_every_n_frames=5)
    # persp_cfg = PerspectiveConfig(mode="every_frame")

    # --- ORIENTATION TOGGLE (edit this one line) ---
    birds_eye_cfg = BirdsEyeConfig(orientation="portrait")
    # birds_eye_cfg = BirdsEyeConfig(orientation="landscape")

    tracker.replace_heatmap_drawer(
        BirdsEyeHeatmapObserver(tracker, birds_eye_cfg, persp_cfg)
    )
    tracker.register_observer(BallCarrierObserver(tracker))
    moves_2_defenderCount_dict, moves_2_timeSincePlayBegan, tracked_video_path, heatmap_video_path, tracked_timeline_map_path = tracker.track_in_each_play()

    # tracker = Tracking(csv_name=csv_name, frame_handler=frame_handler)
    # moves_2_defenderCount_dict, moves_2_timeSincePlayBegan, tracked_video_path, heatmap_video_path, tracked_timeline_map_path = tracker.track_in_each_play()
    del(tracker)

    print_line_to_fill_terminal()
    
    def_count_event_csv_filepath = './results/move_vs_defenders_events.csv'
    def_count_prob_json_filepath = './results/move_policy_by_defenderCount_bin.json'
    time_since_play_event_csv_filepath = './results/move_vs_timeSinceDownFrameCount.csv'
    time_since_play_prob_json_filepath = './results/move_policy_by_frameCountSincePlayStart_bin.json'

    # Defender count: groupby_header = "num_defenders"
    SaveEventDataAndProbabilities(groupby_header="num_defenders",
                                  events_dict=moves_2_defenderCount_dict,
                                  event_csv_filepath=def_count_event_csv_filepath,
                                  prob_json_filepath=def_count_prob_json_filepath)
    
    # Time Since Play Began: groupby_header = "TimeSincePlayBegan"
    SaveEventDataAndProbabilities(groupby_header="TimeSincePlayBegan",
                                  events_dict=moves_2_timeSincePlayBegan,
                                  event_csv_filepath=time_since_play_event_csv_filepath,
                                  prob_json_filepath=time_since_play_prob_json_filepath)

    return tracked_video_path, heatmap_video_path, tracked_timeline_map_path

# For Testing
if __name__ == '__main__':
    process_from_app('#14.csv')
