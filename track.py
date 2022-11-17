import argparse
import pandas as pd
import cv2
import time
import os

from centroid_tracker import (
    CentroidTrackerWithHistory,
    CentroidTrackerWithThreshold
)
from utils import *


def main(video, track_id, save_path):
    cap = cv2.VideoCapture(f'{DATA_PATH}{video}.mkv')
    labels = pd.read_csv(f'{DATA_PATH}{video}.csv')
    
    if video == 3:
        tracker = CentroidTrackerWithThreshold()
    else:
        tracker = CentroidTrackerWithHistory(maxDisappeared=50)
    
    if save_path:
        if type == 'track':
            size = (int(cap.get(3)), int(cap.get(4)))
        else:
            size = (CROP_SIZE, CROP_SIZE)
        
        result = cv2.VideoWriter(
            f'output/{save_path}.mkv', 
            cv2.VideoWriter_fourcc(*'XVID'),
            24.0, 
            size
        )
    frame_ind = 0
    while cap.isOpened():
        time.sleep(0.05)

        ret, frame = cap.read()
        if not ret:
            break

        bboxes = predict(frame_ind, labels)
        bboxes = scale_coords(bboxes, frame)

        if len(bboxes) > 0:
            obj_ids, track_bbs = tracker.update(bboxes)
            if track_id:
                frame = crop(track_id, obj_ids, frame)
            else:
                frame = draw_bbox(frame, track_bbs, bboxes)

        cv2.imshow('frame', frame)

        if save_path:
            result.write(frame)

        frame_ind+=1

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    
    cap.release()
    if save_path:
        result.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object tracking')
    parser.add_argument('--video', type=int, default=1, help='video')
    parser.add_argument('--track_id', type=int, default=None, help='object need to be tracked')
    parser.add_argument('--save_name', type=str, default=None, help='Output video name')

    args = parser.parse_args()
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    main(args.video, args.track_id, args.save_name)