import json
import math
import os
import random

import numpy as np

import cv2
import torch
import torch.nn as nn

import utils
import csv 
import json 

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def csv_to_json(csv_file_path, jsonFilePath):
        jsonArray = []
        #read csv file
        with open(csv_file_path, encoding='utf-8') as csvf: 
            #load csv file data using csv library's dictionary reader
            csvReader = csv.DictReader(csvf) 
            #convert each csv row into python dict
            for row in csvReader: 
                #add this python dict to json array
                jsonArray.append(row)
    
        #convert python jsonArray to JSON String and write to file
        with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
            jsonString = json.dumps(jsonArray, indent=4)
            jsonf.write(jsonString)
            json_file = csv_file_path
            return json_file



class Sign_Data(Dataset):
    def __init__(self,  pose_root, jsonFilePath,  num_samples=25, num_copies=4,
                test_index_file=None):
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        self.data = []
        
        self.test_index_file = test_index_file
        self._make_dataset(pose_root, jsonFilePath) # Use this to relate the classes with labels

        self.pose_root = pose_root

    def __len__(self):
        return len(self.data)
    
    def _make_dataset(self, pose_root, jsonFilePath):
        
        data = self.csv_to_json(self, pose_root, jsonFilePath)
        
        with open(data, 'r') as f:
            content = json.load(f)

        # create label 
        labels = [gloss_entry['labels'] for gloss_entry in content]

        if self.test_index_file is not None:
            print('tested on {}'.format(self.test_index_file))
            with open(self.test_index_file, 'r') as f:
                content = json.load(f)

        # make dataset
        for gloss_entry in content:
            gloss, labels = gloss_entry['gloss'], gloss_entry['label']

            '''
            for instance in instances:
    
                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.data.append(instance_entry)
            '''
