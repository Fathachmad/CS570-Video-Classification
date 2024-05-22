import torch
import torch.utils.data as data_utl
import torchvision.models as models
import torch.utils.data as data_utl
import numpy as np
import cv2
import os
import random
from torchvision import datasets, transforms
import VideoTransform

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

def make_dataset(data_dir, num_classes=3):
    # data_dir : directory of dataset to be augmented
    video_data = os.listdir(data_dir)
    dataset = []
    for vid in video_data:
        cap = cv2.VideoCapture(data_dir+'/'+vid)
        if not cap.isOpened():
            print(f'Error: could not open video file {vid}')
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames)>64:
            frames = frames[:64]
        video_np = np.array(frames)
        n_frames = len(frames)
        if 'Airplane' in data_dir:
            label = 0
        if 'Bird' in data_dir:
            label = 1
        if 'Sky' in data_dir:
            label = 2
        labels = np.zeros((num_classes, n_frames), np.float32)
        labels[label, :] = 1
        dataset.append((video_np, labels, n_frames))
    return dataset
    

class Dataset(data_utl.Dataset):
    def __init__(self, data_dir, transforms=None):

        self.data_dir = data_dir
        self.transforms = transforms
        self.data = make_dataset(self.data_dir)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        
        imgs = self.transforms(vid)
        label = torch.from_numpy(label)
        # return torch.from_numpy(label)
        return video_to_tensor(imgs), label, nf
    
    def __len__(self):
        return len(self.data)
        