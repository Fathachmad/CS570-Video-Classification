import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory of video dataset', default='C:\\Users\\COINSE\Downloads\\tugas\\CS570-Video-Classification\\04 Test_Datasets\\TOTAL_Airplane_within_1km')
parser.add_argument('--batchsize', type=int, help='batch size', default=4)
parser.add_argument('--alpha', type=float, help='beta distribution hyper-parameter alpha', default=8.0)
parser.add_argument('--prob', type=float, help='probability to implement StackMix/TubeMix augmentation', default=0.5)

args = parser.parse_args()

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import VideoTransform
from VideoMix import stackmix
import numpy as np
import cv2
from dataset import Dataset
from moviepy.editor import VideoClip


def custom_collate_fn(batch):
    max_frames = max([x[2] for x in batch])
    
    padded_videos = []
    labels = []
    for video, label, n_frames in batch:
        if n_frames < max_frames:
            pad_size = max_frames - n_frames
            padding = torch.zeros((video.shape[0], pad_size, video.shape[2], video.shape[3]), dtype=video.dtype)
            video = torch.cat((video, padding), axis=1)
            label_padding = torch.zeros((label.shape[0], pad_size), dtype=label.dtype)
            label = torch.cat((label, label_padding), axis=1)
        padded_videos.append(video)
        labels.append(label)
    
    padded_videos = torch.stack(padded_videos, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return padded_videos, labels

def run(data_dir, batch_size=4):
    transform = transforms.Compose([VideoTransform.RandomHorizontalFlip()])
    dataset = Dataset(data_dir, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    idx=0
    for data in dataloader:
        inputs, labels = data
        inputs, labels, cls_labels = stackmix(inputs, labels, args.alpha, args.prob)
        b_size = inputs.size()[0]
        nframes = inputs.size()[2]
        h, w = inputs.size()[3], inputs.size()[4]
        for i in range(b_size):
            filename = f"{data_dir}\\augmented_video_{idx}.mp4"
            fps = 16
            frame = inputs[i].permute(1, 2, 3, 0).cpu().numpy()
            #clip = VideoClip(frame, duration = int(nframes/fps))
            #clip.write_videofile(filename, fps=fps)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videowriter = cv2.VideoWriter(filename, fourcc, fps, [w, h])
            for j in range(nframes):
                frame = inputs[i, :, j, :, :].permute(1, 2, 0).cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                videowriter.write(frame_bgr)
            idx+=1
            print(f"Saved new augmented video: {filename}")
            videowriter.release()
        

if __name__ == '__main__':
    run(args.data_dir, args.batchsize)