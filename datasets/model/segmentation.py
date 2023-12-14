import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchmetrics import Recall, Accuracy, Precision,Dice,JaccardIndex
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import wandb
from U_net import UNet
from torch.optim import lr_scheduler
import numpy as np

def create_dataset():
    DATASET_749 = "Brain-Tumor-Detection/datasets/kaggle_3m"
    dataset_749 = os.listdir(DATASET_749)
    file_paths_image_749 = []
    file_paths_mask_749 = []

    for dir in dataset_749:
        paths_749 = os.path.join(DATASET_749, dir)
        files_749 = os.listdir(paths_749)
        
        for file in files_749:
            if "mask" not in file:
                file_paths_image_749.append(os.path.join(paths_749, file))


    file_paths_mask_749 = [i.split(".")[0]+"_mask.tif" for i in file_paths_image_749]
    

    zip_data=[]
    for idx in range(len(file_paths_image_749)-3700):
        #print(idx)
        image=Image.open(file_paths_image_749[idx])
        mask=Image.open(file_paths_mask_749[idx])
        image_array=torch.Tensor(asarray(image))
        mask_array=torch.Tensor(asarray(mask))
        #print(mask_array.shape,image_array.shape)
        zip_data.append((image_array,mask_array))
    return zip_data

def precision_score_(groundtruth_mask, pred_mask):
    result=0
    for idx in range(pred_mask.shape[0]):
        intersect = np.sum(pred_mask[idx]*groundtruth_mask[idx])
        total_pixel_pred = np.sum(pred_mask[idx])
        precision = np.mean(intersect/total_pixel_pred)
        result+= round(precision, 3)
    return result/pred_mask.shape[0]

def recall_score_(groundtruth_mask, pred_mask):
    result=0
    for idx in range(pred_mask.shape[0]):
        print('Recall ',pred_mask[idx]*groundtruth_mask[idx],(pred_mask[idx]*groundtruth_mask[idx]).shape)
    #     intersect = np.sum(pred_mask[idx]*groundtruth_mask[idx])
    #     total_pixel_truth = np.sum(groundtruth_mask[idx])
    #     recall = np.mean(intersect/total_pixel_truth)
    #     result+= round(recall, 3)
    # return result/pred_mask.shape[0]

def accuracy(groundtruth_mask, pred_mask):
    result=0
    for idx in range(pred_mask.shape[0]):
        intersect = np.sum(pred_mask[idx]*groundtruth_mask[idx])
        union = np.sum(pred_mask[idx]) + np.sum(groundtruth_mask[idx]) - intersect
        xor = np.sum(groundtruth_mask[idx]==pred_mask[idx])
        acc = np.mean(xor/(union + xor - intersect))
        result+= round(acc, 3)
    return result/pred_mask.shape[0]

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3)

def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)

def run_model(
        model:nn.Module,
        epochs:int
):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    zip_data=create_dataset()
    train_data,validate_data,test_data=random_split(zip_data,lengths=[0.7,0.1,0.2],generator=torch.Generator(device=device))
    print(zip_data[0][0].shape,zip_data[0][1].shape)
    train_dataloader=DataLoader(train_data,batch_size=32)
    test_dataloader=DataLoader(test_data,batch_size=32)
    validate_dataloader=DataLoader(validate_data,batch_size=32)
 
    loss_fn= torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=float(0.001))
    scheduler=lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.9)
    result_train={
        'train_loss':[],
        'train_recall':[],
        'train_precision':[],
        'train_dice':[],
        'train_jaccard_index':[],
        'val_loss':[],
        'val_recall':[],
        'val_precision':[],
        'val_dice':[],
        'val_jaccard_index':[],
    }

    result_test={
        'test_loss':[],
        'test_acc':[],
    }

    for epoch in range(1):
        print('Train')
        model.train()
        loss_avg=train(train_dataloader=train_dataloader,validate_dataloader=validate_dataloader,loss_fn=loss_fn,recall_fn=recall_score_,precision_fn=precision_score_,dice_fn=dice_coef,jaccard_index_fn=iou,optimizer=optimizer,model=model,device=device,scheduler=scheduler)
        #result_train['train_dice']=dice_avg
        result_train['train_loss']=loss_avg
        # result_train['train_precision']=precision_avg
        # result_train['train_jaccard_index']=jaccard_index_avg
        # result_train['train_recall']=recall_avg
        #print(f'Epoch {epoch} recall {recall_avg} precision {precision_avg} dice {dice_avg} jaccard_index {jaccard_index_avg}')
        print(f'Epoch {epoch} loss{loss_avg}')

def train(train_dataloader,validate_dataloader,loss_fn,recall_fn,precision_fn,dice_fn,jaccard_index_fn,optimizer,model,device,scheduler):
        loss_avg=0
        recall_avg=0
        precision_avg=0
        dice_avg=0
        jaccard_index_avg=0
        #acc_avg=0
        for batch,(x,y) in enumerate(train_dataloader):
            x,y=x.to(device),y.to(device)
            y_pred=model(x.permute(0,3,2,1))
            print(f'Shape {y_pred.squeeze(1).shape}, y shape {y.shape}')
            y = y.long()
            loss = loss_fn(y_pred.squeeze(1), y.float())
            #recall=recall_fn(y,y_pred.squeeze(1))
            #precision=precision_fn(y.float(),y_pred.squeeze(1))
            #dice=dice_fn(y.float(),y_pred)
            #jaccard_index=jaccard_index_fn(y.float(),y_pred.squeeze(1))
            loss_avg=loss_avg+loss.item()
            #jaccard_index_avg=jaccard_index_avg+jaccard_index.item()
            #precision_avg=precision_avg+precision.item()
            #recall_avg=recall_avg+recall.item()
            #dice_avg=dice_avg+dice.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_avg=loss_avg/len(train_dataloader)
        precision_avg=precision_avg/len(train_dataloader)
        recall_avg=recall_avg/len(train_dataloader)
        dice_avg=dice_avg/len(train_dataloader)
        jaccard_index_avg=jaccard_index_avg/len(train_dataloader)
        scheduler.step()
        validate(dataloader=validate_dataloader,device=device,loss_fn=loss_fn)
        print('Loss avg ', loss_avg)
        return loss_avg

def validate(dataloader,device,loss_fn):
    model.eval()
    with torch.inference_mode():
        loss_avg=0
        for batch,(x,y) in enumerate(dataloader):
            x,y=x.to(device),y.to(device)
            y_pred=model(x.permute(0,3,2,1))
            y = y.long()
            loss = loss_fn(y_pred.squeeze(1), y.float())
            loss_avg=loss_avg+loss.item()
    return loss_avg

model=UNet(3,1)
run_model(model=model,epochs=1)