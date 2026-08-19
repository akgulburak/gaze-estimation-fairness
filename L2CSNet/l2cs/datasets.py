import os
import numpy as np
import cv2

import random

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


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

    def __init__(self, path, root, transform, angle, binwidth, train=True, use_oversample = False, use_class_weights = False):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle

        self.use_oversample = use_oversample
        self.use_class_weights = use_class_weights

        if train==False:
          angle=90
        self.binwidth=binwidth

        self.weights = []
        self.lines = []
        self.demographic_groups = {}

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
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                        sensitive_variable = line.strip().split(" ")[-1]
                        if sensitive_variable not in self.demographic_groups:
                            self.demographic_groups[sensitive_variable] = []
                        self.demographic_groups[sensitive_variable].append(line)

        if use_oversample:
            self.lines = self.apply_oversample(self.lines)

        self.calculate_demographic_group_weights()
    
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        demographic_label = line[-1]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)        
        
        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])

        weight = self.weights[idx]
        
        all_demographic_keys = list(self.demographic_groups.keys())
        if "other" in all_demographic_keys:
            all_demographic_keys.remove("other")
        K = len(all_demographic_keys)

        return img, labels, cont_labels, name, weight, K, demographic_label

class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle,fold=0):
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 42 and abs((label[1]*180/np.pi)) <= 42:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    pitch = label[0]* 180 / np.pi
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    bins = np.array(range(-42, 42,3))
    binned_pose = np.digitize([pitch, yaw], bins) - 1

    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])


    return img, labels, cont_labels, name


