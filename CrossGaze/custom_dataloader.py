'''
    Code based on: https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/datasets.py
'''

import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

import random

class CustomDataset(Dataset):
    def __init__(self, path, root, transform, angle, args, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        self.args = args
        self.train = train
        self.lines = self.read_lines(path)

    def read_lines(self, path):
        all_lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)

                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= self.angle and abs((label[1]*180/np.pi)) <= self.angle:
                        all_lines.append(line)         
        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(all_lines), self.angle))
        return all_lines

    def __len__(self):
        return len(self.lines)

    def __get_data__(self, line):
        line = line.strip().split(" ")

        face = line[0]

        left = line[1]
        
        right = line[2]
        
        name = line[3]
        
        gaze2d = line[5]
        label2d = np.array(gaze2d.split(",")).astype("float")
        label2d = torch.from_numpy(label2d).type(torch.FloatTensor)

        gaze3d = line[4]
        label3d = np.array(gaze3d.split(",")).astype("float")
        label3d = torch.from_numpy(label3d).type(torch.FloatTensor)

        identity = 0
        identity = torch.from_numpy(np.array(identity)).type(torch.LongTensor)

        demographic_label = line[-1]

        return face, left, right, label2d, label3d, name, identity, demographic_label

    def __getinfo__(self, line):
        
        face, left, right, label2d, label3d, name, identity, demographic_label = self.__get_data__(line)
        img = Image.open(os.path.join(self.root, face))

        if self.args.trainer == 'part':
            left_eye_image = Image.open(os.path.join(self.root, left))
            right_eye_image = Image.open(os.path.join(self.root, right))

            transform, eye_transform = self.transform
            if self.transform:
                img = transform(img)
                left_eye_image = eye_transform(left_eye_image)
                right_eye_image = eye_transform(right_eye_image)
                img = [img, left_eye_image, right_eye_image]

        elif self.transform:
            img = self.transform(img)
        
        return img, label2d, label3d, name, identity, demographic_label

    def __getitem__(self, idx):
        line = self.lines[idx]
        img, label2d, label3d, name, identity, demographic_label = self.__getinfo__(line)

        if "weights" in self.__dict__.keys():
            weight = self.weights[idx]
            output = {'img': img, 'label2d': label2d, 'label3d': label3d, 'name': name, 'identity': identity, 'weight': weight, 'K_DG': self.K_DG, 'demographic_label': demographic_label}
        else:
            output = {'img': img, 'label2d': label2d, 'label3d': label3d, 'name': name, 'identity': identity, 'demographic_label': demographic_label}

        return output

class CustomDatasetDemographicGroups(CustomDataset):
    def initialize_demographic_groups(self):
        self.demographic_groups = {}

        for line in self.lines:
            sensitive_variable = line.strip().split(" ")[-1]
            if sensitive_variable not in self.demographic_groups:
                self.demographic_groups[sensitive_variable] = []
            self.demographic_groups[sensitive_variable].append(line)

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

    def __init__(self, path, root, transform, angle, args, train=True, use_reweight=False, use_oversample=False, use_class_weights=False):
        super().__init__(path, root, transform, angle, args, train)

        self.use_reweight = use_reweight
        self.use_oversample = use_oversample
        self.use_class_weights = use_class_weights

        self.lines = self.read_lines(path)
        self.initialize_demographic_groups()
        
        if use_oversample:
            self.lines = self.apply_oversample(self.lines)

        self.weights = []
        self.calculate_demographic_group_weights()

        all_demographic_keys = list(self.demographic_groups.keys())
        self.K_DG = len(all_demographic_keys)