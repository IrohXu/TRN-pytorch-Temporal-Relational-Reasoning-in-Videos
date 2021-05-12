import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from model.TRN import TRN, MultiScaleTRN
from model.LRCN import LRCNs
from utils.dataset import JesterDataset
from utils.train_model import train_model, trans_to_cuda
# from utils.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip, Stack, ToTorchFormatTensor, GroupNormalize

DATA_DIR = './data'
# data_dir = '../dataset'
TRAIN_ANNOTATION_DIR = './label/jester-mini-train.csv'
VAL_ANNOTATION_DIR = './label/jester-mini-validation.csv'
LABELS_DIR = './label/jester-v1-labels.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=DATA_DIR)
parser.add_argument('--train-annotations', default=TRAIN_ANNOTATION_DIR)
parser.add_argument('--val-annotations', default=VAL_ANNOTATION_DIR)
parser.add_argument('--labels', default=LABELS_DIR)
parser.add_argument('--loader-workers', default=2, type=int,
                       help='number of workers for data loading')
parser.add_argument('--num-segs', default=8, type=int,
                       help='number of segmentation for loading dataset')
parser.add_argument('--batch-size', default=5, type=int,
                       help='batch size')
parser.add_argument('--num-frames', default=8, type=int,
                       help='number of frames for TRN')
parser.add_argument('--num-classes', default=27, type=int,
                       help='number of classification results, for Jester it is 27')
parser.add_argument('--img-feature-dim', default=2048, type=int,
                       help='number of CNN avgpool features')
parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay for SGD')
parser.add_argument('--epochs', default=30, type=int,
                        help='number of epochs to train')
parser.add_argument('--model', default='MultiScaleTRN',
                        help='choice of model')

args = parser.parse_args()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__ == '__main__':
    Dataset_train = JesterDataset(data_dir=args.data, annotation_dir=args.train_annotations, labels_dir=args.labels, 
                                    num_segs=args.num_segs, transform=data_transforms['train'])
    dataloader_train = torch.utils.data.DataLoader(Dataset_train, batch_size=args.batch_size,shuffle=True, num_workers=args.loader_workers)
    Dataset_val = JesterDataset(data_dir=args.data, annotation_dir=args.val_annotations, labels_dir=args.labels, 
                                    num_segs=args.num_segs, transform=data_transforms['val'])
    dataloader_val = torch.utils.data.DataLoader(Dataset_val, batch_size=args.batch_size,shuffle=False, num_workers=args.loader_workers)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    dataset_sizes = {'train': len(dataloader_train.dataset), 'val' : len(dataloader_val.dataset)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    if(args.model == 'TRN'):
        model = TRN(args.img_feature_dim, args.num_frames, args.num_segs, args.num_classes)
    elif(args.model == 'MultiScaleTRN'):
        model = MultiScaleTRN(args.img_feature_dim, args.num_frames, args.num_segs, args.num_classes)
    elif(args.model == 'LRCNs'):
        model = LRCNs(args.img_feature_dim, 256, 1, args.num_classes, device)
    else:
        raise("Invalid Model Type")

    print(model)

    model = model.to(device)
    model = trans_to_cuda(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,  momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    model, log = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=args.epochs)
    df=pd.DataFrame({'epoch':[],'training_loss':[],'training_acc':[],'val_loss':[],'val_acc':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['training_acc'] = log['training_acc']
    df['val_loss'] = log['val_loss']
    df['val_acc'] = log['val_acc']
    df.to_csv('training_log_MultiScale.csv',columns=['epoch','training_loss','training_acc','val_loss','val_acc'], header=True,index=False,encoding='utf-8')

    model_save_filename = './best_model.pth'
    torch.save(model.state_dict(), model_save_filename)

