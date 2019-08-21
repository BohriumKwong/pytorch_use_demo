#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:24:24 2019

@author: biototem
"""

from __future__ import print_function, division
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
    model_ft = nn.DataParallel(model_ft)
    
    model_wts = torch.load(model_path)
    model_ft.load_state_dict(model_wts)    
    model_ft = model_ft.to(device)
    return model_ft

def  batch_mil_sampling(imagelist,region_result_npy,mil_data_save_dir,class_name,class_dict,model_ft):
    images = io.imread_collection(imagelist)    
    images = np.stack(images)
#    images_torch = Variable(torch.from_numpy(images.copy().transpose((0,3, 1, 2))).float().div(255).cuda())
    model_ft.eval() 
    with torch.no_grad():
        test_epoch = test.Test_epoch(model_ft,images,128)
        output_predict = test_epoch.predict()
    output_order = np.argsort(output_predict[:,class_dict[class_name]])[::-1]
    if class_dict[class_name] ==0:
        #weight = 0.5
        output_slect = output_order[:int(0.5 * len(output_order))]
    else:
        #weight = 0.2
        output_slect = output_order[int(0.3 * len(output_order)):int(0.5 * len(output_order))]
    for i in range(len(output_slect)):
        os.system('cp ' + imagelist[output_slect[i]] + ' ' + os.path.join(mil_data_save_dir,class_name))
        file_name = os.path.basename(imagelist[output_slect[i]]).split('.')[0]
        # e.g. TCGA-5M-AAT6-01Z-00-DX1-98_6_3_0_M_norm.png to TCGA-5M-AAT6-01Z-00-DX1-98_6_3_0_M_norm
        x = int(file_name.split('-')[-1].split('_')[0])
        # e.g. TCGA-5M-AAT6-01Z-00-DX1-98_6_3_0_M_norm to 98
        y = int(file_name.split('-')[-1].split('_')[1])
        # e.g. TCGA-5M-AAT6-01Z-00-DX1-98_6_3_0_M_norm to 6       
        y_nd = int(file_name.split('-')[-1].split('_')[2])
        # e.g. TCGA-5M-AAT6-01Z-00-DX1-98_6_3_0_M_norm to 3
        y_nd_i = int(file_name.split('-')[-1].split('_')[3])
        region_result_npy[y * y_nd + y_nd_i,x] = int(class_dict[class_name]) + 1

    return region_result_npy       

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
    batch_size = 128
    data_dir_ori = {'train':'/cptjack/totem_disk/totem/M_MSI_MSS/normal/train',
                'val':'/cptjack/totem_disk/totem/M_MSI_MSS/normal/val'}
    
    model_save_path = '/cptjack/totem_disk/totem/kwong/CRC_DC_TRAIN/MIL'
    model_base_name = 'resnet_18'

    model_path = '/cptjack/totem_disk/totem/kwong/CRC_DC_TRAIN/MIL/resnet_18_2.pth'
    model_ft = models.resnet18(pretrained = False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = load_torch_model(model_ft,model_path,device)

    
#    train_epoch = train.Epoch(
#            model_ft, 
#            data_dir, 
#            1, 
#            model_save_path,
#            model_base_name,
#            batch_size
#            )
#    
#    model_wts,train_loss,train_acc,val_loss,val_acc = train_epoch.mil_run()
    
    result_sub = pd.read_csv('/cptjack/totem_disk/totem/kwong/CRC_DC_TRAIN/MIL/TCGA_analysis_statistic_result_0810.csv')
    result_name = sorted(list(result_sub[result_sub['percentage_msi'] + result_sub['percentage_mss']>0].case))
    
    
    classes = sorted(os.listdir(data_dir_ori['train']))
    class_dict = {x:i for i,x in enumerate(sorted(classes))}
    
    region_prediction_dir = '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/M_TCGA_analysis_region_prediction_npy_norm_3/'
    
    mil_sample_result_dir = '/cptjack/totem_disk/totem/sample_result_MIL'
    if not os.path.exists(mil_sample_result_dir):
        os.makedirs(mil_sample_result_dir)
    
    mil_data_save_dir = '/cptjack/totem_disk/totem/MSI_MSS_MIL'

    all_train_loss = [0.6228,0.2192,0.1028,0.2195]
    all_val_loss = [0.5653,0.7438,1.0063,0.6504]#val Loss: 1.0063 Acc:0.7019
    
    all_train_acc = [0.6467,0.9581,0.9783,0.9248]#train Loss: 0.1028 Acc:0.9783
    all_val_acc = [0.7053,0.7042,0.7019,0.72867]
    
    for m in range(3,6):
        # m is the round of MIL processing 
        if os.path.exists(mil_data_save_dir):
            os.system('rm -r ' + mil_data_save_dir)
        print('-' * 16)
        print('Round ' + str(m) + ' is processing: ')
        #model_path = '/cptjack/totem_disk/totem/kwong/CRC_DC_TRAIN/MIL/resnet_18_'+ str(m-1) + '.pth'
        #model_ft = load_torch_model(model_path,device)

        for class_name in classes: 
            if not os.path.exists(os.path.join(mil_data_save_dir,class_name)):
                os.makedirs(os.path.join(mil_data_save_dir,class_name))
            for image_name in result_name:
                imagelist = glob.glob(os.path.join(data_dir_ori['train'],class_name,'*' + image_name.split('.')[0] + '*'))
                if len(imagelist) > 0:
                    preview_filename = image_name + '-region_3_class_output.npy'
                    if os.path.exists(region_prediction_dir + preview_filename):
                        region_result_npy = np.load(region_prediction_dir + preview_filename)
                        region_result_npy[region_result_npy !=3] =0
                        try:
                            region_result_npy = batch_mil_sampling(imagelist,region_result_npy,mil_data_save_dir,class_name,class_dict,model_ft)
                            np.save(os.path.join(mil_sample_result_dir,image_name.split('.')[0] + '_' + str(m) + '_sresult.npy') ,region_result_npy)
                        except Exception:
                            print('Round ' + str(m) + '_' + image_name + ' error!')
        
        print('Round ' + str(m) + ' begin training: ')

        data_dir = {'train':mil_data_save_dir,
                'val':'/cptjack/totem_disk/totem/M_MSI_MSS/normal/val'}
        
        train_epoch = train.Epoch(
                model_ft, 
                data_dir,
                m, 
                model_save_path,
                model_base_name,
                batch_size
                )
        try:        
            model_ft,train_loss,train_acc,val_loss,val_acc = train_epoch.mil_run()                            
            all_train_acc.append(train_acc)
            all_train_loss.append(train_loss)
            all_val_acc.append(val_acc)
            all_val_loss.append(val_loss)      
        except Exception:
            print('Round ' + str(m) + ' training error!')
    np.save('all_train_acc.npy', all_train_acc)
    np.save('all_train_loss.npy', all_train_loss)
    np.save('all_val_acc.npy', all_val_acc)
    np.save('all_val_loss.npy', all_val_loss)
    
    
    plt.plot(all_train_acc,c = 'red',label = 'train_acc')
    plt.plot(all_val_acc,c = 'blue',label = 'val_acc')
    plt.legend()
    plt.show()
    
    plt.plot(all_train_loss,c = 'red',label = 'train_loss')
    plt.plot(all_val_loss,c = 'blue',label = 'val_loss')
    plt.legend()
    plt.show()
