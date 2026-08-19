from torch.utils.data.dataset import Dataset

import os
import glob

import cv2
import numpy as np
import random

import torch
from torchvision.transforms.functional import to_tensor

class Gaze360(Dataset):
    def calculate_demographic_group_weights(self):
        number_of_total_samples = sum(len(s) for s in self.demographic_groups.values())

        group_weights = dict.fromkeys(self.demographic_groups.keys(), 0)
        for group_key in self.demographic_groups:
            if group_key == "other":
                group_weights[group_key]=1
            else:
                group_weights[group_key] = number_of_total_samples / len(self.demographic_groups[group_key])

        for line in self.lines:
            demographic_group = line.split(" ")[-1].strip().lstrip()
            self.weights.append(group_weights[demographic_group]) 

    def apply_oversample(self, lines):
        demographic_group_dict = {}
        for line in lines:
            demographic_group = line.strip().split(" ")[-1]
            if demographic_group not in demographic_group_dict:
                demographic_group_dict[demographic_group] = []
            demographic_group_dict[demographic_group].append(line)

        majority_group_length = max((len(s) for s in demographic_group_dict.values()), default=0)
        
        for group_name in demographic_group_dict.keys():
            if group_name != "other":
                added_samples = random.choices(demographic_group_dict[group_name], k = majority_group_length - len(demographic_group_dict[group_name]))
                lines.extend(added_samples)
        return lines

    def __init__(self, data, use_oversample, use_regularize):
        self.image_root_path = data.image
        self.label_root_path = data.label

        self.use_oversample = use_oversample
        self.use_regularize = use_regularize

        self.lines = []
        self.weights = []

        self.demographic_groups = {}
        
        with open(self.label_root_path) as f:
            lines = f.readlines()
            lines.pop(0)
            self.orig_list_len = len(lines)
            for line in lines:
                self.lines.append(line)

        if use_oversample:
            self.lines = self.apply_oversample(self.lines)

        for line in self.lines:
            gaze2d = line.strip().split(" ")[5]
            label = np.array(gaze2d.split(",")).astype("float")

            sensitive_variable = line.strip().split(" ")[-1]
            ethnicity = line.strip().split(" ")[-1]
            if sensitive_variable not in self.demographic_groups:
                self.demographic_groups[sensitive_variable] = []
            self.demographic_groups[sensitive_variable].append(line)
                
        self.calculate_demographic_group_weights()

    def __len__(self):
        print(len(self.lines))
        exit()
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        # lefteye = line[1]
        # righteye = line[2]
        # name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        img = cv2.imread(os.path.join(self.image_root_path, face))
        img = to_tensor(img)
        data = {"face": img}
        data["weight"] = self.weights[idx]
        return data, label