#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:26:00 2019

@author: root
"""

import torch.nn as nn
from torchvision import models
import os,sys
import matplotlib.pyplot as plt
import numpy as np
from utils import train,test
import torch
from skimage import io
sys.path.append('/cptjack/totem_disk/totem/kwong/EfficientNet-PyTorch/')
from efficientnet_pytorch import EfficientNet


os.environ["CUDA_VISIBLE_DEVICES"] = "2"



if __name__ == '__main__':
    model_ft_normal =  EfficientNet.from_name('efficientnet-b4')   
    num_ftrs = model_ft_normal._fc.in_features
    model_ft_normal._fc = nn.Linear(in_features= num_ftrs, out_features = 2,bias=True)

    model_path = '/cptjack/totem_disk/totem/colon_pathology_data/train/save/efficientnet-b4_epoch_0.pth'
    model_ft_normal = nn.DataParallel(model_ft_normal)
    model_ft_normal.load_state_dict(torch.load(model_path))
    
    folder_name = '/cptjack/totem_disk/totem/colon_pathology_data/batch_4/ETL/train'
    batch_size = 32
    test_epoch = test.Test_epoch_from_folder(model_ft_normal,folder_name,batch_size)
    result = test_epoch.predict()