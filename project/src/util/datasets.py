# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    # for imagenet dataset
    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    # dataset = datasets.ImageFolder(root, transform=transform)
    
    #for cifar-100 dataset
    root = os.path.join(args.data_path, 'train' if is_train else 'test')
    dataset = datasets.CIFAR100(root, train=is_train, transform=transform, download=True)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    #ImageNet default mean and std
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD

    #Cifar-100 mean and std
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform for imagenet
    # t = []
    # if args.input_size <= 224:
    #     crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    # size = int(args.input_size / crop_pct)
    # t.append(
    #     transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    # )
    # t.append(transforms.CenterCrop(args.input_size))

    # t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(mean, std))

    #eval transform for cifar-100
    t = []
    if args.input_size <= 32:
        crop_pct = 32 / 32  # 32x32
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(t)
