import torch
from torch import nn
import pandas as pd
import numpy as np

import sys
import cv2
import os
import os.path
import pickle
from tqdm import tqdm
sys.path.append('./custom_yolo/')
from customyolo import CustomYOLO

models_list = ['binary_augmented_ft', 'full_augmented_ft']
label_idx_list = [(0, 1), (0, 1)] # (plane, bird)

for model_path, label_idx in zip(models_list, label_idx_list):
    print('Woking on model {}'.format(model_path))

    for (root, dirs, files) in os.walk('../../data/datasets/'):
        # No file in directory
        if len(files) == 0:
            continue

        print('Working on directory {}'.format(root))
        
        # Formatting save path
        save_path1 = 'data_augmented' if 'augmented' in root else 'data'
        save_path2 = 'eval' if 'Test' in root else 'train'
        if 'Airplane' in root:
            save_path3 = 'airplane'
        elif 'Bird' in root:
            save_path3 = 'bird'
        else:
            save_path3 = 'clear'

        # Create track directory
        save_dir = '../../data/data/{}/{}/{}/{}'.format(model_path, save_path1, save_path2, save_path3)
        os.makedirs(save_dir, exist_ok=True)

        for file in tqdm(files):
            # Pickle file exists
            if os.path.isfile('{}/{}'.format(save_dir, file + '.pt')):
                continue

            if '.mp4' in file:
                max_id = 0 # Maximum value of track id
                idx = 0 # idx for current frame
                track_id_dict = {} # track_id_dict[id] = {[assigned_id (int), last_idx (int)]}
                track_dict = {} # track_dict[id] = [(conf1, conf2, xn, yn, xn, yn), ...] (List)

                model = CustomYOLO('../../data/models/{}.pt'.format(model_path))
                cap = cv2.VideoCapture(root + '/' + file)
                    
                while cap.isOpened():
                    # Read a frame from the video
                    success, frame = cap.read()
                    
                    # Successfully read the frame
                    if success:                    
                        result = model.track(source=frame, conf=0.01, tracker="./custom_tracker.yaml", persist=True, verbose=False)
                        boxes = result[0].boxes.xyxyn.cpu()
                        idx = idx + 1

                        # No object detected
                        if len(boxes) == 0 or result[0].boxes.id is None or result[0].tot_conf is None:
                            continue

                        # Track ids, labels, confidences for each boxes
                        track_ids = result[0].boxes.id.int().cpu().tolist()
                        tot_conf = result[0].tot_conf.cpu()

                        # Some data missing
                        if not (len(boxes) == len(track_ids) and len(boxes) == len(tot_conf)):
                            continue

                        for box, track_id, conf in zip(boxes, track_ids, tot_conf):
                            conf0 = float(conf[label_idx[0]])
                            conf1 = float(conf[label_idx[1]])

                            # Not enough confidence
                            if conf0 < 0.01 and conf1 < 0.01:
                                continue

                            # Given track_id is never added or skipped
                            if (track_id not in track_id_dict) or track_id_dict[track_id][1] != idx - 1:
                                max_id = max_id + 1
                                track_id_dict[track_id] = [max_id, idx]
                                track_dict[max_id] = [[conf0, conf1, float(box[0]), float(box[1]), float(box[2]), float(box[3])]]

                            # Given track_id is alive
                            else:
                                track_id_dict[track_id][1] = idx
                                track_dict[track_id_dict[track_id][0]].append([conf0, conf1, float(box[0]), float(box[1]), float(box[2]), float(box[3])])

                    # Failed to read the frame
                    else:
                        break
                
                cap.release()
                cv2.destroyAllWindows()

                # No object detected
                if len(track_dict) == 0:
                    with open('{}/{}'.format(save_dir, file + '.pt'), 'wb') as pck:
                        pass
                    continue

                max_len = len(max(track_dict.values(), key=lambda x:len(x)))
                track_len = []
                track_data = []

                for track in track_dict.values():
                    track_len.append(len(track))
                    track_data.append(track)

                    for i in range(max_len - len(track)):
                        track_data[-1].append([0, 0, 0, 0, 0, 0])

                track_len_tensor = torch.tensor(track_len)
                track_data_tensor = torch.tensor(np.array(track_data))
                
                with open('{}/{}'.format(save_dir, file + '.pt'), 'wb') as pck:
                    pickle.dump(track_len_tensor, pck)
                    pickle.dump(track_data_tensor, pck)
