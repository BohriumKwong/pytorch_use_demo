# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:40:36 2019

@author: Bohrium.Kwong
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
#import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm as tqdm
from tqdm._tqdm import trange
from torchvision import datasets, transforms
import os,sys
import gc
import time
from sklearn.metrics import balanced_accuracy_score,recall_score
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Test_epoch_from_array:
    r"""
    :param model: a model that loaded with pytorch and trans .to(device)
    :param input_array: a 4D numpy array [n,h,w,c] which includes n images transform to numpy array(RGB mode)
    :param batch_size: the value of batch_size in model's predicting
    :param verbose: show the information of tdqm method with dataloaders in processing or not
    """      
    def __init__(self, model, input_array,batch_size=256,mean = [0,0,0], std = [1,1,1],verbose=False):      
        self.model = model
        self.input_array = input_array
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.verbose = verbose  
        
        
    def predict(self): 
        self.input_array /= 255
        for i in range(3):
            self.input_array[:,:,:,i] -= self.mean[i]
            self.input_array[:,:,:,i] /= self.std[i]
        if self.input_array.shape[0] <= self.batch_size :
            image_tensor = Variable(torch.from_numpy(self.input_array.copy().transpose((0,3, 1, 2))).float().cuda())
            output =  self.model(image_tensor) 
            output_predict = F.softmax(output)
            output_predict = output_predict.cuda().data.cpu().numpy()
        else:
            batch_count = self.input_array.shape[0]// self.batch_size
#            with tqdm(dataloaders[phase], desc = phase, file=sys.stdout, disable=not (self.verbose)) as iterator: 
            for i in trange(batch_count + 1,disable=not (self.verbose)):
                top = i * self.batch_size
                bottom = min(self.input_array.shape[0], (i+1) * self.batch_size)
                if top < self.input_array.shape[0]:
                    image_tensor = Variable(torch.from_numpy(self.input_array[top:bottom,:,:,:].copy().transpose((0,3, 1, 2))).float().cuda())
                    output =  self.model(image_tensor) 
                    output_predict_batch = F.softmax(output)
                    output_predict_batch = output_predict_batch.cuda().data.cpu().numpy()
                    if i ==0 :
                        output_predict = output_predict_batch.copy()
                    else:
                        output_predict = np.row_stack((output_predict,output_predict_batch.copy()))
                    del image_tensor,output_predict_batch,output
                    gc.collect()
        return output_predict
    
    
    
class Test_epoch_from_folder:
    r"""
    :param model: a model that loaded with pytorch and trans .to(device)
    :param folder_path: the sub directories are  like 'path/lable true name/*.{png,jpg,tif,bmp, ...}':
    :param batch_size: the value of batch_size in model's predicting
    :param verbose: show the information of tdqm method with dataloaders in processing or not
    :param if_plot: plot the confusion_matrix of the result of summarize
    """ 
    def __init__(self, model, folder_path,batch_size,mean = [0,0,0], std = [1,1,1],verbose=True,if_plot = True):  
       self.model = model
       self.folder_path = folder_path
       self.batch_size = batch_size
       self.mean = mean
       self.std = std
       self.verbose = verbose
       self.if_plot = if_plot


    def get_dataloader(self):
        data_transforms = transforms.Compose([ transforms.ToTensor(),transforms.Normalize(self.mean,self.std)])
    
        image_datasets = datasets.ImageFolder(self.folder_path,data_transforms)
        
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = self.batch_size, shuffle = True, num_workers=0)
        #一般默认num_workers为1，但如果是在服务器上的docker中运行训练代码时，batch size设置得过大，shared memory不够(因为docker限制了shm)
        # 解决方法是将num_workers设置为0
        class_names = image_datasets.classes

        classFileCount = np.array([sum([len(files) for root, dirs, files in os.walk(os.path.join(self.folder_path,class_name))]) \
                  for class_name in class_names])
        weight_bias = classFileCount.max()/classFileCount
        #此处返回的是根据样本类别数量自适应生成的loss加权
        
        return dataloaders,weight_bias,class_names

    def predict(self): 
        dataloaders,weight_bias,class_names = self.get_dataloader()    
        loss_function  = nn.CrossEntropyLoss(weight=torch.from_numpy(weight_bias).float().to(device))
#        optimizer = optim.Adam(self.model.parameters(),
#                            lr=0.000001, weight_decay=0.0001)
        self.model.eval()
        result = np.zeros((1,len(class_names)))
        running_loss = 0.0
        running_corrects = 0
        logs = {}
        batch_count = 0
        time_start = time.time()
        with tqdm(dataloaders, desc = os.path.basename(self.folder_path), file=sys.stdout, disable=not (self.verbose)) as iterator: 
            for inputs, labels in iterator:
                inputs = inputs.to(device)
                labels = labels.to(device)                            
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                region_result = preds.data.cpu().numpy()
                loss = loss_function(outputs, labels)
                loss_logs = {'CrossEntropyLoss': loss.item()}
                logs.update(loss_logs)
                running_loss += loss.item()
                labels = labels.data.cpu().numpy()
                
                balanced_acc = balanced_accuracy_score(labels,region_result)
                recall = recall_score(labels,region_result,average='weighted')                                
                metrics_meters = {'Balanced_acc': balanced_acc,'recall':recall}
                logs.update(metrics_meters)
                running_corrects += balanced_acc
                if self.verbose:
                    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
                    s = ', '.join(str_logs)
                    iterator.set_postfix_str(s)
                                
                batch_count += 1
                result = np.row_stack((result, np.column_stack((labels,region_result))))
                        
            epoch_loss = running_loss/ batch_count
            epoch_acc = running_corrects / batch_count
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(os.path.basename(self.folder_path), epoch_loss, epoch_acc))
            
        if self.if_plot:
            print('\n')
            print(classification_report(result[:,0], result[:,1],target_names = class_names))
            print('\n')
            cm = confusion_matrix(result[:,0], result[:,1])
            plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
            plt.title('Confusion matrix of ' + os.path.basename(self.folder_path), size=15)
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, class_names,rotation=45, size=10)
            plt.yticks(tick_marks, class_names,size=10)
            plt.tight_layout()
            plt.ylabel('Actual label',size=15)
            plt.xlabel('Predicted label',size=15)
            width, height=cm.shape
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(cm[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')
            plt.show()

        time_elapsed = time.time() - time_start
        print('testing complete in {:.0f}m {:.0f}s'.format(time_elapsed //60, time_elapsed %60))
        
        return result            
                
        
               