import json
import math
import os
import random

import numpy as np

import cv2
import torch
import torch.nn as nn

import utils
from ast import literal_eval
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def compute_difference(x):
    diff = []

    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)

        diff.append(temp)

    return diff

def read_pose_file(filepath, video_id):

    try:        content = json.load(open(filepath))
    except IndexError:
        return None
    
    vid = content[video_id]["labels"]
    id = content[video_id]["id"]

    save_to = os.path.join('../../code/TGCN/features', vid)

    try:
        ft = torch.load(os.path.join(save_to, id + '_ft.pt'))

        xy = ft[:, :2]
        return xy, vid

    except FileNotFoundError:
        
        indexPIP_0_X = content[0]["indexPIP_0_X"]
        indexPIP_0_Y = content[0]["indexPIP_0_Y"]
        indexPIP_1_X = content[0]["indexPIP_1_X"]
        indexPIP_1_Y = content[0]["indexPIP_1_Y"]
        
        leftElbow_Y = content[0]["leftElbow_Y"]
        leftElbow_X = content[0]["leftElbow_X"]
        
        ringPIP_0_Y = content[0]["ringPIP_0_Y"]
        ringPIP_0_X = content[0]["ringPIP_0_X"]
        ringPIP_1_X = content[0]["ringPIP_1_X"]
        ringPIP_1_Y = content[0]["ringPIP_1_Y"]
        
        
        thumbMP_0_X = content[0]["thumbMP_0_X"]
        thumbMP_1_Y = content[0]["thumbMP_1_Y"]
        thumbMP_1_X = content[0]['thumbMP_1_X']
        thumbMP_0_Y = content[0]['thumbMP_0_Y']
  
        
        ringDIP_1_X = content[0]['ringDIP_1_X']
        ringDIP_1_Y = content[0]['ringDIP_1_Y']
        ringDIP_0_X = content[0]['ringDIP_0_X']
        ringDIP_0_Y = content[0]['ringDIP_0_Y']
        
        leftEye_Y = content[0]['leftEye_Y']
        leftEye_X = content[0]['leftEye_X']
        
        littleMCP_1_X = content[0]['littleMCP_1_X']
        littleMCP_1_Y = content[0]['littleMCP_1_Y']
        littleMCP_0_X = content[0]['littleMCP_0_X']
        littleMCP_0_Y = content[0]['littleMCP_0_Y']
        
        middleDIP_0_X = content[0]['middleDIP_0_X']
        middleDIP_1_X = content[0]['middleDIP_1_X']
        middleDIP_0_Y = content[0]['middleDIP_0_Y']
        middleDIP_1_Y = content[0]['middleDIP_1_Y']

        ringTip_0_X = content[0]['ringTip_0_X']
        ringTip_1_X = content[0]['ringTip_1_X']
        ringTip_1_Y = content[0]['ringTip_1_Y']
        ringTip_0_Y = content[0]['ringTip_0_Y']

        rightEye_X = content[0]['rightEye_X']
        rightEye_Y = content[0]['rightEye_Y']
        
        wrist_1_Y = content[0]['wrist_1_Y']
        wrist_1_X = content[0]['wrist_1_X']
        wrist_0_X = content[0]['wrist_0_X']
        wrist_0_Y = content[0]['wrist_0_Y']
        
        littleTip_1_Y = content[0]['littleTip_1_Y']
        littleTip_1_X = content[0]['littleTip_1_X']
        littleTip_0_Y = content[0]['littleTip_0_Y']
        littleTip_0_X = content[0]['littleTip_0_X']
        
        thumbTip_0_X = content[0]['thumbTip_0_X']
        thumbTip_0_Y = content[0]['thumbTip_0_Y']
        thumbTip_1_X = content[0]['thumbTip_1_X']
        thumbTip_1_Y = content[0]['thumbTip_1_Y']
    
        middleMCP_1_Y = content[0]['middleMCP_1_Y']
        middleMCP_0_X = content[0]['middleMCP_0_X']
        middleMCP_1_X = content[0]['middleMCP_1_X']
        middleMCP_0_Y = content[0]['middleMCP_0_Y']
        
        rightWrist_Y = content[0]['rightWrist_Y']
        rightWrist_X = content[0]['rightWrist_X']

        middleTip_1_Y = content[0]['middleTip_1_Y']
        middleTip_0_Y = content[0]['middleTip_0_Y']
        middleTip_0_X = content[0]['middleTip_0_X']
        middleTip_1_X = content[0]['middleTip_1_X']

        ringMCP_0_Y = content[0]['ringMCP_0_Y']
        ringMCP_1_X = content[0]['ringMCP_1_X']
        ringMCP_1_Y = content[0]['ringMCP_1_Y']
        ringMCP_0_X = content[0]['ringMCP_0_X']

        thumbCMC_1_X = content[0]['thumbCMC_1_X']
        thumbCMC_0_Y = content[0]['thumbCMC_0_Y']
        thumbCMC_1_Y = content[0]['thumbCMC_1_Y']
        thumbCMC_0_X = content[0]['thumbCMC_0_X']
        
        indexDIP_1_Y = content[0]['indexDIP_1_Y']
        indexDIP_0_Y = content[0]['indexDIP_0_Y']
        indexDIP_1_X = content[0]['indexDIP_1_X']
        indexDIP_0_X = content[0]['indexDIP_0_X']
        
        neck_Y = content[0]['neck_Y']
        neck_X = content[0]['neck_X']

        middlePIP_1_Y = content[0]['middlePIP_1_Y']
        middlePIP_0_Y = content[0]['middlePIP_0_Y']
        middlePIP_0_X = content[0]['middlePIP_0_X']
        middlePIP_1_X = content[0]['middlePIP_1_X']

        indexTip_1_X = content[0]['indexTip_1_X']
        indexTip_0_Y = content[0]['indexTip_0_Y']
        indexTip_1_Y = content[0]['indexTip_1_Y']
        indexTip_0_X = content[0]['indexTip_0_X']

        rightShoulder_Y = content[0]['rightShoulder_Y']
        rightShoulder_X = content[0]['rightShoulder_X']

        indexMCP_0_X = content[0]['indexMCP_0_X']
        indexMCP_1_X = content[0]['indexMCP_1_X']
        indexMCP_1_Y = content[0]['indexMCP_1_Y']
        indexMCP_0_Y = content[0]['indexMCP_0_Y']
        
        nose_Y = content[0]['nose_Y']
        nose_X = content[0]['nose_X']

        thumbIP_0_X = content[0]['thumbIP_0_X']
        thumbIP_1_Y = content[0]['thumbIP_1_Y']
        thumbIP_0_Y = content[0]['thumbIP_0_Y']
        thumbIP_1_X = content[0]['thumbIP_1_X']

        rightEar_X = content[0]['rightEar_X']
        rightEar_Y = content[0]['rightEar_Y']
   
        leftEar_X = content[0]['leftEar_X']
        leftEar_Y = content[0]['leftEar_Y']

        leftShoulder_Y = content[0]['leftShoulder_Y']
        leftShoulder_X = content[0]['leftShoulder_X']


        leftWrist_X = content[0]['leftWrist_X']
        leftWrist_Y = content[0]['leftWrist_Y']

        rightElbow_X= content[0]['rightElbow_X']
        rightElbow_Y = content[0]['rightElbow_Y']
    
        littlePIP_1_X = content[0]['littlePIP_1_X']
        littlePIP_1_Y = content[0]['littlePIP_1_Y']
        littlePIP_0_X = content[0]['littlePIP_0_X']
        littlePIP_0_Y = content[0]['littlePIP_0_Y']
   

        littleDIP_1_Y = content[0]['littleDIP_1_Y']
        littleDIP_1_X = content[0]['littleDIP_1_X']
        littleDIP_0_X = content[0]['littleDIP_0_X']
        littleDIP_0_Y = content[0]['littleDIP_0_Y']

        # 
        x = literal_eval(indexPIP_0_X) + literal_eval(indexPIP_1_X) + literal_eval(leftElbow_X) + literal_eval(ringPIP_1_X) + literal_eval(ringPIP_0_X) + literal_eval(littleDIP_0_X) + literal_eval(littleDIP_1_X) + literal_eval(littlePIP_0_X) + literal_eval(littlePIP_1_X) + literal_eval(rightElbow_X) + literal_eval(leftWrist_X) + literal_eval(leftShoulder_X) + literal_eval(leftEar_X) + literal_eval(nose_X) + literal_eval(rightEar_X) + literal_eval(thumbIP_0_X) + literal_eval(thumbIP_1_X) +  literal_eval(rightShoulder_X)# + literal_eval(indexMCP_0_X) + literal_eval(indexMCP_1_X) + literal_eval(indexTip_1_X) + literal_eval(indexTip_0_X) + literal_eval(thumbMP_0_X) + literal_eval(thumbMP_1_X)# + literal_eval(ringDIP_1_X) +  literal_eval(ringDIP_0_X) + literal_eval(leftEye_X) + literal_eval(littleMCP_1_X) + literal_eval(littleMCP_0_X) + literal_eval(middleDIP_0_X) + literal_eval(middleDIP_1_X) + literal_eval(ringTip_0_X) + literal_eval(ringTip_1_X) + literal_eval(rightEye_X) + literal_eval(wrist_1_X) + literal_eval(wrist_0_X) + literal_eval(littleTip_1_X) + literal_eval(littleTip_0_X) +  literal_eval(neck_X) + literal_eval(middlePIP_0_X) + literal_eval(middlePIP_1_X) + literal_eval(indexDIP_0_X) + literal_eval(indexDIP_1_X) + literal_eval(thumbCMC_0_X) + literal_eval(thumbCMC_1_X) + literal_eval(ringMCP_0_X) + literal_eval(ringMCP_1_X) +  literal_eval(middleTip_1_X) + literal_eval(middleTip_0_X) + literal_eval(thumbTip_0_X) + literal_eval(thumbTip_1_X) + literal_eval(middleMCP_0_X) + literal_eval(middleMCP_1_X) + literal_eval(rightWrist_X) 
        #  
        y = literal_eval(indexPIP_0_Y) + literal_eval(indexPIP_1_Y) + literal_eval(leftElbow_Y) + literal_eval(ringPIP_1_Y) + literal_eval(ringPIP_0_Y) + literal_eval(littleDIP_0_Y) + literal_eval(littleDIP_1_Y) + literal_eval(littlePIP_0_Y) + literal_eval(littlePIP_1_Y) + literal_eval(rightElbow_Y) + literal_eval(leftWrist_Y) + literal_eval(leftShoulder_Y) + literal_eval(leftEar_Y) + literal_eval(nose_Y) + literal_eval(rightEar_Y) + literal_eval(thumbIP_0_Y) + literal_eval(thumbIP_1_Y) +  literal_eval(rightShoulder_Y)# + literal_eval(indexMCP_0_Y) + literal_eval(indexMCP_1_Y) + literal_eval(indexTip_1_Y) + literal_eval(indexTip_0_Y) + literal_eval(thumbMP_0_Y) + literal_eval(thumbMP_1_Y)# + literal_eval(ringDIP_1_Y) +  literal_eval(ringDIP_0_Y) + literal_eval(leftEye_Y) + literal_eval(littleMCP_1_Y) + literal_eval(littleMCP_0_Y) + literal_eval(middleDIP_0_Y) + literal_eval(middleDIP_1_Y) + literal_eval(ringTip_0_Y) + literal_eval(ringTip_1_Y) + literal_eval(rightEye_Y) + literal_eval(wrist_1_Y) + literal_eval(wrist_0_Y) + literal_eval(littleTip_1_Y) + literal_eval(littleTip_0_Y) +  literal_eval(neck_Y) + literal_eval(middlePIP_0_Y) + literal_eval(middlePIP_1_Y) + literal_eval(indexDIP_0_Y) + literal_eval(indexDIP_1_Y) + literal_eval(thumbCMC_0_Y) + literal_eval(thumbCMC_1_Y) + literal_eval(ringMCP_0_Y) + literal_eval(ringMCP_1_Y) +  literal_eval(middleTip_1_Y) + literal_eval(middleTip_0_Y) + literal_eval(thumbTip_0_Y) + literal_eval(thumbTip_1_Y) + literal_eval(middleMCP_0_Y) + literal_eval(middleMCP_1_Y) + literal_eval(rightWrist_Y) 
        
        x_diff = torch.FloatTensor(compute_difference(x)) / 2
        y_diff = torch.FloatTensor(compute_difference(y)) / 2

        zero_indices = (x_diff == 0).nonzero()

        orient = y_diff / x_diff
        orient[zero_indices] = 0
        
        xs = torch.Tensor(x)
        ys = torch.Tensor(y)

        xy = torch.stack([xs, ys]).transpose_(0, 1)

        ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

        vid = content[video_id]["labels"]
        id = content[video_id]["id"]
        save_to = os.path.join('../../code/TGCN/features', vid)
        # save_to = os.path.join('../../code/TGCN/features', vid)
        if not os.path.exists(save_to):
            os.mkdir(save_to)
        torch.save(ft, os.path.join(save_to, id + '_ft.pt'))

        xy = ft[:, :2]

        return xy , vid

class Sign_Dataset(Dataset):
    def __init__(self, index_file_path, pose_root, num_samples, test_index_file=None):
        assert os.path.exists(index_file_path), "Non-existent indexing file path: {}.".format(index_file_path)
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        self.test_index_file = test_index_file
        self._make_dataset(index_file_path)
        self.num_samples = num_samples
        self.index_file_path = index_file_path
        self.pose_root = pose_root


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gloss_cat, video_id = self.data[index]
        # frames of dimensions (T, H, W, C)
        x, y = self._load_poses(video_id)
        # y = gloss_cat

        return x, y, video_id

    def _make_dataset(self, index_file_path):
        with open(index_file_path, 'r') as f:
            content = json.load(f)

        # create label encoder
        #glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])
        glosses = [gloss_entry['gloss'] for gloss_entry in content]

        self.label_encoder.fit(glosses)
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))

        if self.test_index_file is not None:
            with open(self.test_index_file, 'r') as f:
                content = json.load(f)

        # make dataset
        for gloss_entry in content:
            gloss, labels = gloss_entry['gloss'], gloss_entry['label']
            instance_entry = gloss , labels
            self.data.append(instance_entry)

    def _load_poses(self, video_id):
        """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
         """
        poses = []
        # pose_path = os.path.join(self.pose_root, video_id, self.framename.format(str(i).zfill(5)))
        pose_path = os.path.join(self.pose_root)
        # pose = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
        pose, vid = read_pose_file(pose_path, video_id)
        poses.append(pose)
        poses_across_time = torch.cat(poses, dim=1)
        return poses_across_time , vid


