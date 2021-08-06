from __future__ import print_function, division
import os
import torch

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import cv2


from PIL import Image

def GetSamplerWeights(text_file):
    class_balance = { "COVID-19": 0, "normal": 0, "pneumonia":0}
    for line in open(text_file).readlines():
        cla = line.split()[2]
        class_balance[cla]+=1

    weights = []
    for line in open(text_file).readlines():
        cla = line.split()[2]                          
        weights.append(1/class_balance[cla])
    
    return weights

class COVID_DATASET(Dataset):
    def __init__(self, text_file, data_dir,mapping, transform = None):
        
        
        print('Generating Dataset')
        print('data directory: ', data_dir)
        print('text file: ', text_file)
        
        self.file_list = open(text_file).readlines()
        self.data_dir = data_dir
        self.transform = transform
        self.mapping = mapping
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(self.file_list[idx])
        img_name = self.file_list[idx].split()[1]
        #print('img name:', img_name)
        class_name = self.file_list[idx].split()[2]
        #print('class:',class_name)
       
        
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.fromarray(cv2.imread(img_path))
        #print(type(img))
        #print(img.shape)
        label = self.mapping[class_name]
        
        sample = {'image': img, 'label': label, 'file_name': img_path}
        
        if self.transform:
            t_img = self.transform(sample['image'])
            sample = {'image': t_img, 'label': label, 'file_name': img_path}
        return sample

    
#custom transforms ===========
class robbie_croptop_transform(object):
    def __init__(self, percentage = .08):
        assert isinstance(percentage, float)
        self.percentage = percentage
    def __call__ (self,image):

        image = np.asarray(image)

        offset = int(image.shape[0] * self.percentage)
        cropped_image = image[offset:]
        
        
        return Image.fromarray(cropped_image)
    

    
        
class robbie_central_crop(object):
    def __init__(self):
        print()
    def __call__ (self, image):
        image = np.asarray(image)
        size = min(image.shape[0], image.shape[1])
        offset_h = int((image.shape[0] - size) / 2)
        offset_w = int((image.shape[1] - size) / 2)
        cropped_image = image[offset_h:offset_h + size, offset_w:offset_w + size]
        return Image.fromarray(cropped_image)
    
class robbie_resize(object):
    def __init__(self, size):
        self.size = size
    def __call__ (self, image):
        image = np.asarray(image)
        resized_image = cv2.resize(image, (self.size,self.size))
        return Image.fromarray(resized_image)
    
class robbie_rand_ratio_resize(object):

    def __init__(self, prob = .3, delta = .1):
        self.prob = prob
        self.delta = delta
        
    def __call__(self, image):
        if np.random.rand()>=self.prob:
            return image
        image = np.asarray(image)
        
        ratio = image.shape[0]/image.shape[1]
        ratio = np.random.uniform(max(ratio - self.delta, 0.01), ratio + self.delta)
        
        if ratio * image.shape[1] <= image.shape[1]:
            size = (int(image.shape[1] * ratio), image.shape[1])
        else:
            size = (image.shape[0], int(image.shape[0] / ratio))
            
        dh = image.shape[0] - size[1]
        top, bot = dh // 2, dh - dh // 2
        dw = image.shape[1] - size[0]
        left, right = dw // 2, dw - dw // 2
        
        if size[0] > 224 or size[1] > 224:
            print(image.shape, size, ratio)
        
        image = cv2.resize(image, size)
        image = cv2.copyMakeBorder(image, top, bot, left, right,cv2.BORDER_CONSTANT,(0, 0, 0))
        
        if image.shape[0] != 224 or image.shape[1] != 224:
            raise ValueError(img.shape, size)
        return Image.fromarray(image)
    
    
class robbie_norm(object):
    def __init__(self):
        print()
    def __call__ (self,image):
        
        return torch.div(image, 255)

