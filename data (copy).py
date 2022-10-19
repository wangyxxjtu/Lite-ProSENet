import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random

from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate

import pandas
import SimpleITK as sitk


class DataBowl3Classifier(Dataset):
    def __init__(self, root_path, phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')

        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []

        self.filenames = [os.path.join(root_path, idx) for idx in os.listdir(root_path)]

        label_path = "/data/yujwu/NSCLC/NSCLC_Radiomic_ Lung_version3_201911.csv"
        data_annotations = np.array(pandas.read_csv(label_path))
        self.names = data_annotations[:, 0]
        self.surv_time = data_annotations[:, 8]

    def data_norm(self, data):
        data_max = np.max(data)
        data_min = np.min(data)
        inter = data_max - data_min
        data = data - data_min
        data = (data/inter*255)
        return data

    def __getitem__(self, idx, split=None):
        filename = self.filenames[idx] # get the data
        # get the position of GTV
        pos = (filename.upper()).find("GTV")
        # get the name before GTV
        name = filename[pos-9: pos]
        # get the index of name in self.names
        index = np.where(self.names == name)
        # print(index)
        # get the survival year based on the index
        Thresh = 3
        Survival_time = int(self.surv_time[index]/365.0 + 0.5)

        if Survival_time > Thresh:
            label = 1
        else:
            label = 0
        # label = int(self.surv_time[index]/365.0 + 0.5)
        # read the data of filename
        data = sitk.ReadImage(filename)
        # get the 3D image
        img = sitk.GetArrayFromImage(data)

        img = self.data_norm(img)

        return img, label

    def __len__(self):
        sample_num = len(self.filenames)
        return sample_num

