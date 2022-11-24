import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import mea_model

def run():

    transform_train = transforms.Compose([
                #randomresizedcrop for cifar-100:
                
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #mean and std for Cifar-100 dataset:
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
            

    dataset_train = datasets.CIFAR100(root='../../../', train=True, download=True, transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=16,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )

    model = mea_model.__dict__['mae_vit_base_patch16'](norm_pix_loss=True)
    model.to('cuda')
    model_without_ddp = model
    print(model)


    param_groups = optim_factory.add_weight_decay(model_without_ddp, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=0.01, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    from engine_pretrain import train_one_epoch
    device = torch.device('cuda')
    epoch = 0

    model.train()

    optimizer.zero_grad()

    for images, _ in data_loader_train:
        images = images.to(device)
        print(images.shape)
        with torch.cuda.amp.autocast():
            loss, _, _ = model(images, mask_ratio=0.75)
    
if __name__ == '__main__':
    run()
