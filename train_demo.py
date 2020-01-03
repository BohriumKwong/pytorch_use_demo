#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 14:05:13 2019

@author: biototem
"""

import torch
import torch.nn as nn
from torchvision import models
import os,sys
import matplotlib.pyplot as plt
import numpy as np
from utils import train,test
import pandas as pd
import glob
from skimage import io
sys.path.append('/cptjack/totem_disk/totem/kwong/EfficientNet-PyTorch/')
from efficientnet_pytorch import EfficientNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_torch_model(model_ft,model_path,device):
    for param in model_ft.parameters():
        param.requires_grad = False    
    param_update=[]
    for name,param in model_ft.layer3.named_parameters():
        param.requires_grad = True
        param_update.append(param)
    for name,param in model_ft.layer4.named_parameters():
        param.requires_grad = True
        param_update.append(param)
    for name,param in model_ft.fc.named_parameters():
        param.requires_grad = True
        param_update.append(param)
#    model_ft = nn.DataParallel(model_ft)
    
    model_wts = torch.load(model_path)
    model_ft.load_state_dict(model_wts) 
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)
    return model_ft


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    device_ids = [int(0),int(1)]        
    batch_size = 256
    data_dir_ori = {'train':'/cptjack/totem_disk/totem/colon_pathology_data/batch_4/ETL/train',
                'val':'/cptjack/totem_disk/totem/colon_pathology_data/batch_4/ETL/val'}
    
    model_save_path = '/cptjack/totem_disk/totem/colon_pathology_data/train/save/res18-hue'
    model_base_name = 'resnet-18'

#    model_path = './resnet_18_2.pth'
    
#    model_ft =  EfficientNet.from_name('efficientnet-b4')
    model_ft = models.resnet18(num_classes=2,pretrained = False)
#    num_ftrs = model_ft._fc.in_features
#    model_ft._fc = nn.Linear(in_features= num_ftrs, out_features = 2,bias=True)
#    model_ft._fc = nn.Linear(in_features=1536, out_features=2, bias=True)
#    model_ft = load_torch_model(model_ft,model_path,device)
    
    model_ft = model_ft.cuda()
    model_ft = nn.DataParallel(model_ft)
    train_epoch = train.Epoch(
            model_ft, 
            data_dir_ori, 
            1, 
            model_save_path,
            model_base_name,
            batch_size
            )
    
    model_wts,train_loss,train_acc,val_loss,val_acc = train_epoch.run(10,3,True)
    
    plt.figure()
    plt.style.use('bmh') # bmh
    plt.title("training acc and loss: ", fontsize=18)
    train_acc, = plt.plot(train_acc,c = 'g',linewidth=3)
    val_acc, = plt.plot(val_acc,c = 'b',linewidth=3)
    train_loss, = plt.plot(train_loss,c = 'y',linewidth=3)
    val_loss, = plt.plot(val_loss,c = 'r',linewidth=3)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel('acc-loss')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(handles=[train_acc, val_acc,train_loss,val_loss], labels=['train_acc', 'val_acc','train_loss','val_loss'],
        loc='center right') 
    plt.show()
    plt.close()