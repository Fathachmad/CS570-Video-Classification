import torch
from torch import nn

from torchinfo import summary

import sys
import cv2
sys.path.append('./custom_yolo/')
from customyolo import CustomYOLO
from ultralytics import YOLO

model = CustomYOLO('birds_planes_200e.pt')

video_path = "../data/plane.mp4"
cap = cv2.VideoCapture(video_path)

# Maximum value of track id
max_id = 0
# track_id_dict[id] = {[alive (bool), assigned_id (int)]}
track_id_dict = {}
# track_dict[id] = [(conf1, conf2, xn, yn, xn, yn), ...] (List)
track_dict = {}

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # Successfully read the frame
    if success:
        result = model.track(source=frame, conf=0.01, tracker="custom_track.yaml", persist=True)
        boxes = result[0].boxes.xyxyn.cpu()

        # No object detected
        if len(boxes) == 0 :
            continue
        
        names = result[0].names

        # Track ids, labels, confidences for each boxes
        track_ids = result[0].boxes.id.int().cpu().tolist()
        labels = result[0].boxes.cls.int().cpu()
        confs = result[0].tot_conf.cpu()

        for box, track_id, label, conf in zip(boxes, track_ids, labels, confs):
            # Given track_id is never added or not alive
            if (track_id not in track_id_dict) or not track_id_dict[track_id][0]:
                max_id = max_id + 1
                track_id_dict[track_id] = (True, max_id)
                track_dict[max_id] = [float(conf[0]), float(conf[1]), float(box[0]), float(box[1]), float(box[2]), float(box[3])]

            # Given track_id is alive
            else:
                track_dict[track_id_dict[track_id][1]].append([float(conf[0]), float(conf[1]), float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        
    # Failed to read the frame
    else:
        break
    
print(track_dict.keys())

"""
"""


cap.release()
cv2.destroyAllWindows()

class FreezeModule(nn.Module):
    def __init__(self, module):
        super(FreezeModule, self).__init__()
        self.module = module
        
    def forward(self, inp):
        results = model.track(source=inp, tracker="bytetrack.yaml")
        return x