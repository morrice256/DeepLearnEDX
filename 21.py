#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:24:43 2018

@author: morrice256
"""
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from torchvision import transforms
from dataset import Datasetcustom
train_csv_file='training_labels.csv'
validation_csv_file='validation_labels.csv'
train_dataset = Datasetcustom(csv_file=train_csv_file 
                        , data_dir='training_data_pytorch/')
validation_data = Datasetcustom(csv_file=validation_csv_file
                          , data_dir='validation_data_pytorch/')

samples = [53, 23, 10]
for x in samples:
    item = train_dataset.__getitem__(x)
    plt.imshow(item[0])
    plt.show()
    
samples2 = [22, 32, 45]
for x in samples2:
    item = validation_data.__getitem__(x)
    plt.imshow(item[0])
    plt.show()
    
    
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
      
test_normalize = Datasetcustom(csv_file=train_csv_file 
                        , data_dir='training_data_pytorch/',
                        transform = composed)

print("Mean: ", test_normalize[0][0].mean(dim = 1).mean(dim = 1))
print("Std:", test_normalize[0][0].std(dim = 1).std(dim = 1))