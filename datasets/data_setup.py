import os
from pathlib import Path
import glob
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
import torchvision.transforms as transforms
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# import tensorflow as tf
import numpy as np
import cv2 

def preprocessing():

    BATCH_SIZE=32
    transform =  transforms.Compose([
            transforms.Resize([256,256]), 
            transforms.ToTensor(),
        ])
    
    #datasets
    train_dataset=ImageFolder(root='Brain-Tumor-Detection/datasets/segmentation_dataset/train',transform=transform)
    test_dataset=ImageFolder(root='Brain-Tumor-Detection/datasets/segmentation_dataset/test',transform=transform)

    #dataloaders
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE)

    return train_dataloader,test_dataloader

preprocessing()

def create_dir(dir_name_main:str):
    if os.path.isdir(dir_name_main):
        print('Directory already exists')
    else:
        path=Path('Brain-Tumor-Detection/datasets/'+dir_name_main)
        os.makedirs(path,exist_ok=True)
        
        #train yes,bo,pred
        path_train_no=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/train/no')
        os.makedirs(path_train_no,exist_ok=True)
        path_train_yes=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/train/yes')
        os.makedirs(path_train_yes,exist_ok=True)
        path_train_pred=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/train/pred')
        os.makedirs(path_train_pred,exist_ok=True)

        #test yes,bo,pred
        path_test_no=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/test/no')
        os.makedirs(path_test_no,exist_ok=True)
        path_test_yes=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/test/yes')
        os.makedirs(path_test_yes,exist_ok=True)
        path_train_pred=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/test/pred')
        os.makedirs(path_train_pred,exist_ok=True)
        print('Creating directory')


def create_dataset(paths:[],dir_name_main:str):
    create_dir(dir_name_main)
    for path in paths:
        print(path)
        for subdir, dirs, files in os.walk(path):
            for dir_name in dirs:
                print(dir_name)
                if dir_name=='no':
                        print('if',dir_name)
                        for subdir, dirs, files in os.walk(path+'/'+dir_name):
                            #print('Files',subdir, dirs, files)
                            dp = IterableWrapper(range(len(files)))
                            train, test = dp.random_split(total_length=len(files), weights={"train": 0.8, "test": 0.2}, seed=42)
                            print('Train',list(train))
                            for file in files:
                                print('Image',file)
                                #print('Condition',re.findall('\d+',file)!=0 and int(re.findall('\d+',file)[0]) in train)
                                if len(re.findall('\d+',file))!=0 and int(re.findall('\d+',file)[0]) in train:
                                    os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/train/no/{file}")
                                else:
                                    os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/test/no/{file}")
                elif dir_name=='yes':
                        for subdir, dirs, files in os.walk(path+'/'+dir_name):
                                dp = IterableWrapper(range(len(files)))
                                train, test = dp.random_split(total_length=len(files), weights={"train": 0.8, "test": 0.2}, seed=42)
                                for file in files:
                                    if len(re.findall('\d+',file))!=0 and int(re.findall('\d+',file)[0]) in train:
                                        os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/train/yes/{file}")
                                    else:
                                        os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/test/yes/{file}")
                elif dir_name=='pred':
                        for subdir, dirs, files in os.walk(path+'/'+dir_name):
                                dp = IterableWrapper(range(len(files)))
                                train, test = dp.random_split(total_length=len(files), weights={"train": 0.8, "test": 0.2}, seed=42)
                                for file in files:
                                    if len(re.findall('\d+',file))!=0 and int(re.findall('\d+',file)[0]) in train:
                                        os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/train/pred/{file}")
                                    else:
                                        os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/test/pred/{file}")


#create_dataset(paths=['Brain-Tumor-Detection/datasets/brain_tumor_dataset','Brain-Tumor-Detection/datasets/Brain_Tumor_Detection'],dir_name_main='segmentation_dataset')