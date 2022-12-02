import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import mea_model as models_mae

import random
import argparse

imagenet_mean = np.array([0.5071, 0.4867, 0.4408])
imagenet_std = np.array([0.2675, 0.2565, 0.2761])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, rn, chkpt_num):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.50)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    # plt.subplot(1, 4, 1)
    show_image(x[0], "original")
    os.makedirs(f'../data/{chkpt_num}/{rn}', exist_ok=True)
    
    plt.savefig(f'../data/{chkpt_num}/{rn}/original.png')
    plt.show()

    # plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")
    plt.savefig(f'../data/{chkpt_num}/{rn}/masked.png')
    plt.show()
    # plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")
    plt.savefig(f'../data/{chkpt_num}/{rn}/reconstruction.png')
    plt.show()
    # plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")
    plt.savefig(f'../data/{chkpt_num}/{rn}/reconstruction_visible.png')
    plt.show()

if __name__ == '__main__':
    # img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
    img_urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer6.png', 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog2.png', 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship7.png', 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane7.png']
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--chkpt_dir', type=str)

    args = parser.parse_args()

    chkpt_num = args.chkpt_dir.split('/')[-1].split('.')[0].split('-')[-1]
        
    model = prepare_model(args.chkpt_dir)

    for img_url in img_urls:
        img = Image.open(requests.get(img_url, stream=True).raw)
        img = img.resize((32, 32))
        img = np.array(img) / 255.

        assert img.shape == (32, 32, 3)

        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        plt.rcParams['figure.figsize'] = [5, 5]
        show_image(torch.tensor(img))
        plt.show()
        # generate a random number  
        r = random.randint(0, 1000)
       

        run_one_image(img, model, r, chkpt_num)


