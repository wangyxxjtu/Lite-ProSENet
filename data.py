import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import  PIL
import os
from PIL import Image
import time
import collections
import random

from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate

import pandas
import SimpleITK as sitk

from scipy.ndimage.interpolation import rotate

def augment(sample,  ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        angle1 = np.random.rand() * 180
        size = np.array(sample.shape[2:4]).astype('float')
        rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                           [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
        sample = rotate(sample, angle1, axes=(2, 1), reshape=False)
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[0]:
            axisorder = np.random.permutation(2)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))

    if ifflip:
        flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[::flipid[0], ::flipid[1], ::flipid[2]])
    return sample

class DataBowl3Classifier(Dataset):
    def __init__(self, root_path, phase='train', isAugment=True):
        assert (phase == 'train' or phase == 'val' or phase == 'test')

        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []

        self.filenames = [os.path.join(root_path, idx) for idx in os.listdir(root_path)]

        self.filenames = list(filter(lambda item: 'GTV' in item,self.filenames))
        # print(len(list(self.filenames)))

        # label_path = "/home/yujwu/Data/NSCLC/survival_estimate/survival_est_1.6/NSCLC_Radiomic_ Lung_version3_201911.csv"
        label_path = "data/NSCLC_PROCESSED.CSV"

        data_frame = pandas.read_csv(label_path)
        data_frame['age'] = data_frame['age'].fillna(68)/100

        # data_annotations = np.array(pandas.read_csv(label_path))
        data_annotations = np.array(data_frame)
        self.names = data_annotations[:,1]
        self.surv_time = data_annotations[:,29]
        self.event = data_annotations[:,30]
        self.clinical = data_annotations[:,2:29].astype('float')
        self.isAugment = isAugment

    def data_resample(self,image, isAugment=True):
        new_x_size = 96
        new_y_size = 96
        new_z_size = 12


        new_size = [new_x_size, new_y_size, new_z_size]
        new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                       zip(image.GetSize(), image.GetSpacing(), new_size)]

        if isAugment:
            imgarray = np.array(sitk.GetArrayFromImage(image))
            augtype = {'flip': True, 'swap': False, 'rotate': False, 'scale': False}
            aug_img_array = augment(imgarray,
                                  ifflip=augtype['flip'], ifrotate=augtype['rotate'],
                                  ifswap=augtype['swap'])
            augimge=sitk.GetImageFromArray(aug_img_array)
            image = augimge

        interpolator_type = sitk.sitkLinear
        new_img = sitk.Resample(image, new_size, sitk.Transform(), interpolator_type, image.GetOrigin(), new_spacing,
                                image.GetDirection(), 0.0, image.GetPixelIDValue())
        return new_img

    def data_norm(self, data):
        data_max = np.max(data)
        data_min = np.min(data)
        inter = data_max - data_min
        data = data - data_min
        data = (data/inter*255)
        # transforms.Resize((224, 224))
        # data = np.reshape(data,(210,10))
        return data

    def __getitem__(self, idx, split=None):
        filename = self.filenames[idx] # get the data
        # get the position of GTV
        pos = (filename.upper()).find("GTV")
        # pos = filename.find("GTV")
        # get the name before GTV
        name = filename[pos-9: pos]
        # get the index of name in self.names
        index = np.where(self.names == name)
        # print(index)
        # get the survival year based on the index
        # Thresh = 3
        # # Survival_time = int(self.surv_time[index]/365.0 + 0.5)
        #
        # if Survival_time > Thresh:
        #     label = 1
        # else:
        #     label = 0
        label = (self.surv_time[index][0])/max(self.surv_time)
        # label = int(self.surv_time[index]/365.0 + 0.5)
        # read the data of filename
        data = sitk.ReadImage(filename)
        # get the 3D image
        data = self.data_resample(data, self.isAugment)
        # print(f"dimention:{data.GetSize()}")
        img = sitk.GetArrayFromImage(data)

        img = self.data_norm(img)
        event = self.event[index][0]
        if event == 1:
            event = True
        elif event==0:
            event = False
        # event = self.event[index]
        # return img, label,event
        clinical = self.clinical[index][0]
        # print(clinical,'cliniacl')
        return img, np.diag(clinical), label, event

    def __len__(self):
        sample_num = len(self.filenames)
        return sample_num

