# -*- coding: utf-8 -*-
#
import sys
sys.path.append('/home/aistudio/work/Pix2Vox-F')
import numpy as np
import os, cv2
import random
import paddle
from visualdl import LogWriter

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger


# Set up data augmentation
IMG_SIZE = 224,224
CROP_SIZE = 128,128
train_transforms = utils.data_transforms.Compose([
    utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
    # utils.data_transforms.RandomBackground([[225, 255], [225, 255], [225, 255]]),
    # utils.data_transforms.ColorJitter(.4, .4, .4),
    # utils.data_transforms.RandomNoise(.1),
    # utils.data_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # utils.data_transforms.RandomFlip(),
    # utils.data_transforms.RandomPermuteRGB(),
    # utils.data_transforms.ToTensor(),
])

rendering_images = []
root = '/home/aistudio/work/Pix2Vox-F/input'
for image_path in os.listdir(root):
    print(os.path.join(root, image_path))
    if 'png' not in os.path.join(root, image_path):
        continue
    rendering_image = cv2.imread(os.path.join(root, image_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    if len(rendering_image.shape) < 3:
        print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                (dt.now(), image_path))
        sys.exit(2)

    rendering_images.append(rendering_image)
print(len(rendering_images))
cv2.imwrite('rendering_image.png', rendering_images[1]*255)
rendering_images = train_transforms(np.asarray(rendering_images))
print(rendering_images.shape) # (1, 224, 224, 4)
cv2.imwrite('transforms.png', rendering_images[1]*255)
