# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:41:48 2019

@author: Bohrium.Kwong
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from tqdm import tqdm as tqdm
from torchvision import datasets, transforms
import os,sys
import gc
import time
from sklearn.metrics import balanced_accuracy_score,recall_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Epoch:
    r"""
    :param model: a model that loaded with pytorch and trans .to(device)
    :param datapath: a dick that defined as {'train':train dataset path,'val':val dataset path}
    :param mil_round: the value of epoch in MIL training method
    :param model_save_path: the path of saving trained model 
    :param model_base_name: the base name that used in saving trained model
    :param batch_size: the value of batch_size in training model
    :param verbose: show the information of tdqm method with dataloaders in processing or not 
    """  
    
    def __init__(self, model, datapath, mil_round,model_save_path,model_base_name,batch_size,verbose=True):      
        self.model = model
        self.datapath = datapath
        self.mil_round = mil_round
        self.model_save_path = model_save_path
        self.model_base_name = model_base_name
        self.batch_size = batch_size
        self.verbose = verbose       


    def get_dataloader(self):
        data_transforms = {
        'train':transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ]),
        'val':transforms.Compose([ transforms.ToTensor()
        ]),}
    
        image_datasets = {x: datasets.ImageFolder(self.datapath[x],
                                          data_transforms[x])
                for x in ['train', 'val']}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = self.batch_size, shuffle = True, num_workers=0)
                for x in ['train', 'val']}
        #一般默认num_workers为1，但如果是在服务器上的docker中运行训练代码时，batch size设置得过大，shared memory不够(因为docker限制了shm)
        # 解决方法是将num_workers设置为0
        class_names = image_datasets['train'].classes

        classFileCount = np.array([sum([len(files) for root, dirs, files in os.walk(os.path.join(self.datapath['train'],class_name))]) \
                  for class_name in class_names])
        weight_bias = classFileCount.max()/classFileCount
        #此处返回的是根据样本类别数量自适应生成的loss加权
        
        return dataloaders,weight_bias

    def mip_run(self):       
        dataloaders,weight_bias = self.get_dataloader()    
        loss_function  = nn.CrossEntropyLoss(weight=torch.from_numpy(weight_bias).float().to(device))
        optimizer = optim.Adam(self.model.parameters(),
                            lr=0.000001, weight_decay=0.0001)
#        train_loss = []
#        val_loss = []
#    
#        train_acc = []
#        val_acc = []
#        time_start = time.time()
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase=='train':
                #scheduler.step()
                self.model.train()
            else:
                self.model.eval()
            running_loss = 0.0
            running_corrects = 0
            logs = {}
            batch_count = 0
            with tqdm(dataloaders[phase], desc = phase, file=sys.stdout, disable=not (self.verbose)) as iterator:            
                for inputs, labels in iterator:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        region_result = preds.data.cpu().numpy()
                        loss = loss_function(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    loss_logs = {'CrossEntropyLoss': loss.item()}
                    logs.update(loss_logs)
                    running_loss += loss.item()
                    
                    balanced_acc = balanced_accuracy_score(labels.data.cpu().numpy(),region_result)
                    recall = recall_score(labels.data.cpu().numpy(),region_result,average='weighted')
                    metrics_meters = {'Balanced_acc': balanced_acc,'recall':recall}
                    logs.update(metrics_meters)
                    running_corrects += balanced_acc
                    if self.verbose:
                        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
                        s = ', '.join(str_logs)
                        iterator.set_postfix_str(s)
                        
                    batch_count += 1
                
            epoch_loss = running_loss/ batch_count
            epoch_acc = running_corrects / batch_count
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase=='train':
                train_loss = epoch_loss
                train_acc =  epoch_acc
                best_model_wts = copy.deepcopy(self.model).state_dict()
                torch.save(best_model_wts, os.path.join(self.model_save_path,self.model_base_name + '_' +str(self.mil_round)+'.pth'))
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
                
            
        return self.model,train_loss,train_acc,val_loss,val_acc
        
    def run(self,num_epochs=30,early_patience=5):       
        dataloaders,weight_bias = self.get_dataloader()    
        loss_function  = nn.CrossEntropyLoss(weight=torch.from_numpy(weight_bias).float().to(device))
        optimizer = optim.Adam(self.model.parameters(),
                            lr=0.000001, weight_decay=0.0001)
        train_loss = []
        val_loss = []
    
        train_acc = []
        val_acc = []
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        min_loss = 0.8
        early_stop_flag = 0
        stop_print = False
        
        for epoch in range(num_epochs):
            if early_stop_flag <= early_patience:
                #check early_stopping or not
                    
                time_start = time.time()
                print("Epoch {}/{}".format(epoch,num_epochs -1))
                print('-'*10)
                for phase in ['train', 'val']:
                    if phase=='train':
                        #scheduler.step()
                        self.model.train()
                    else:
                        self.model.eval()
                    running_loss = 0.0
                    running_corrects = 0
                    logs = {}
                    batch_count = 0
                    with tqdm(dataloaders[phase], desc = phase, file=sys.stdout, disable=not (self.verbose)) as iterator:            
                        for inputs, labels in iterator:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            
                            optimizer.zero_grad()
                            
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = self.model(inputs)
                                _, preds = torch.max(outputs, 1)
                                region_result = preds.data.cpu().numpy()
                                loss = loss_function(outputs, labels)
                                
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                            loss_logs = {'CrossEntropyLoss': loss.item()}
                            logs.update(loss_logs)
                            running_loss += loss.item()
                            
                            balanced_acc = balanced_accuracy_score(labels.data.cpu().numpy(),region_result)
                            recall = recall_score(labels.data.cpu().numpy(),region_result,average='weighted')
                            metrics_meters = {'Balanced_acc': balanced_acc,'recall':recall}
                            logs.update(metrics_meters)
                            running_corrects += balanced_acc
                            if self.verbose:
                                str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
                                s = ', '.join(str_logs)
                                iterator.set_postfix_str(s)
                                
                            batch_count += 1
                        
                    epoch_loss = running_loss/ batch_count
                    epoch_acc = running_corrects / batch_count
                    
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    if phase=='train':
                        train_loss.append(epoch_loss)
                        train_acc.append(epoch_acc)
                    else:
                        val_loss.append(epoch_loss)
                        val_acc.append(epoch_acc)  
                         

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = copy.deepcopy(epoch_acc)
                    elif phase == 'val' and epoch > 0  and epoch_loss < min_loss:
                        min_loss = copy.deepcopy(epoch_loss)
                        #deep copy the model & min_loss and saved the best model
                        best_model_wts = copy.deepcopy(self.model).state_dict()
                        torch.save(best_model_wts, os.path.join(self.model_save_path,self.model_base_name + \
                                                                time.strftime('_%Y_%m_%d_.pth',time.localtime(time.time()))))
                        del best_model_wts
                        gc.collect()
                        early_stop_flag = 0
                    elif phase == 'val' and epoch > 0 and epoch_loss > min_loss:
                        early_stop_flag = early_stop_flag + 1   
                        # Let early_stop_flag itself adds by 1, which is used by early_stopping 
                    print()
                if ~stop_print:
                    print("Epoch early stop: {}/{}".format(epoch,num_epochs -1))
                    stop_print = True


                
        time_elapsed = time.time() - time_start
        print('training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed //60, time_elapsed %60))
        print('Best test Acc: {:.4f} Loss: {:.4f}'.format(best_acc,min_loss))
        
        return self.model,train_loss,train_acc,val_loss,val_acc
        