import numpy as np
import cv2


DATA_PATH = 'data/test_video_'
CROP_SIZE = 416
DATA_FOLDER = 'output'


def predict(frame_ind, labels):
    detections = labels[labels['frame']==frame_ind]
    detections = detections[['x0', 'y0', 'x1', 'y1']].to_numpy()
    return detections


def scale_coords(bboxes, frame):
    frame_size = frame.shape
    res = []
    for bbox in bboxes:
        x1, x2 = int(bbox[0] * frame_size[1]), int(bbox[2] * frame_size[1])
        y1, y2 = int(bbox[1] * frame_size[0]), int(bbox[3] * frame_size[0])
        res.append([x1, y1, x2, y2])
    return np.array(res)


def draw_bbox(frame, track_bbs, bboxes):
    for idx, j in track_bbs.items():
        if j.tolist() not in bboxes.tolist():
            continue
        x1, y1, x2, y2 = int(j[0]), int(j[1]), int(j[2]), int(j[3])
        name_idx = f'car: {idx}'

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        frame = cv2.putText(frame, name_idx, (x1+(x2-x1)//2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 4)

    return frame


def crop(track_id, objcts, frame):
    h, w = frame.shape[0:2]
    for idx, j in objcts.items():
        if idx == track_id:
            width, height = j
            
            half = CROP_SIZE//2

            if height - (half+1) <= 0:
                low = 1
                up = half * 2 + 1
            elif height + (half+1) >= h:
                up = h - 1
                low = h - half * 2 - 1
            else:
                up = height + half
                low = height - half

            
            if width - (half + 1) <= 0:
                left = 1
                right = half * 2 + 1
            elif width + (half + 1) >= w:
                right = w - 1
                left = w - half * 2 - 1
            else:
                right = width + half
                left = width - half
                        
            frame = frame[low:up, left:right]
    return frame
