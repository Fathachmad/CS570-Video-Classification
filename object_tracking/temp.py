import torch
import os
import sys
import pickle
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from random import shuffle

sys.path.append('./custom_yolo/')
from customyolo import CustomYOLO
"""
models = ['birds_planes_200e', 'full_dataset_100e', 'detect_best']
augmented = [False, True]
train = [False, True]"""

module = 'birds_planes_200e'
augmented = True
train = True
length = []
track = []
label = []
label_list = []  # For computing class weight

for (root, dirs, files) in os.walk('../../data/data/{}/{}/{}/'.format(module,
                                                                      'data_augmented' if augmented else 'data',
                                                                      'train' if train else 'eval')):
    for file in files:
        file_path = os.path.join(root, file)
        label_path = root[root.rfind('/'):]

        # Append label data
        if 'airplane' in label_path:
            label.append(torch.tensor([1, 0, 0], dtype=torch.float64))
            label_list.append(0)
            print("Label : airplane")
        elif 'bird' in label_path:
            label.append(torch.tensor([0, 1, 0], dtype=torch.float64))
            label_list.append(1)
            print("Label : bird")
        else:
            label.append(torch.tensor([0, 0, 1], dtype=torch.float64))
            label_list.append(2)
            print("Label : clear sky")

        # No object detected, as clear sky
        if os.path.getsize(file_path) == 0:
            length.append(None)
            track.append(None)
            label_list.pop()  # It's not trainable, so don't consider for training
            continue

        # Track data exsists
        with open(file_path, 'rb') as pck:
            print(pickle.load(pck))
