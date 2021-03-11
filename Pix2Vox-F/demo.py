#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import cv2
import sys
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint

from config import cfg

import json
import paddle
import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.voxel

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    
    parser.add_argument('--imgs', dest='imgs', help='demo dir', default=None)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args

def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights

    # Print config
    print('Demo Use Config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if args.imgs:
        demo_net(cfg, args.imgs)


def demo_net(cfg, imgs_path):
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    merger = Merger(cfg)

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    encoder_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "encoder.pdparams"))
    encoder.set_state_dict(encoder_state_dict)
    decoder_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "decoder.pdparams"))
    decoder.set_state_dict(decoder_state_dict)

    if cfg.NETWORK.USE_MERGER:
        merger_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "merger.pdparams"))
        merger.set_state_dict(merger_state_dict)

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    merger.eval()

    rendering_images = []
    if os.path.isfile(imgs_path):
        print("demo img")
        rendering_image = cv2.imread(imgs_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        rendering_image = np.asarray(rendering_image)[np.newaxis,:,:,:]
        # print(rendering_image.shape)
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        rendering_image = test_transforms(rendering_image)
        # print(rendering_image)
        rendering_image = paddle.reshape(rendering_image, [1, 1, 3, 224, 224])
        with paddle.no_grad():
            # Get data from data loader
            rendering_image = utils.network_utils.var_or_cuda(rendering_image)

            # Test the encoder, decoder and merger
            image_features = encoder(rendering_image)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = paddle.mean(generated_volume, axis=1)

            for th in cfg.TEST.DEMO_VOXEL_THRESH:
                _volume = paddle.greater_equal(generated_volume, paddle.to_tensor(th)).astype("float32")
                _volume = paddle.reshape(_volume, [32, 32, 32])
                # print(_volume.shape)
                # print(_volume)
                # Append generated volumes to TensorBoard
                if cfg.DIR.OUT_PATH:
                    # Volume Visualization
                    pred_file_name = os.path.join(cfg.DIR.OUT_PATH, imgs_path.split('/')[-1].split('.')[0]+'.obj')
                    print("save ",pred_file_name)
                    utils.voxel.voxel2obj(pred_file_name, _volume.cpu().numpy())

    elif os.path.isdir(imgs_path):
        print("demo dir")
        rendering_files_path = os.listdir(imgs_path)
        for rendering_file_path in rendering_files_path:
            if '.png' not in rendering_file_path:
                continue
            print(os.path.join(imgs_path, rendering_file_path))
            rendering_image = cv2.imread(os.path.join(imgs_path, rendering_file_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            rendering_image = np.asarray(rendering_image)[np.newaxis,:,:,:]
            # print(rendering_image.shape)
            IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
            CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
            test_transforms = utils.data_transforms.Compose([
                utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
                utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
                utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
                utils.data_transforms.ToTensor(),
            ])

            rendering_image = test_transforms(rendering_image)
            # print(rendering_image)
            rendering_image = paddle.reshape(rendering_image, [1, 1, 3, 224, 224])
            with paddle.no_grad():
                # Get data from data loader
                rendering_image = utils.network_utils.var_or_cuda(rendering_image)

                # Test the encoder, decoder and merger
                image_features = encoder(rendering_image)
                raw_features, generated_volume = decoder(image_features)

                if cfg.NETWORK.USE_MERGER:
                    generated_volume = merger(raw_features, generated_volume)
                else:
                    generated_volume = paddle.mean(generated_volume, axis=1)

                # for th in cfg.TEST.VOXEL_THRESH:
                #     _volume = paddle.greater_equal(generated_volume, paddle.to_tensor(th)).astype("float32")
                #     print(_volume.shape)

                # Append generated volumes to TensorBoard
                if cfg.DIR.OUT_PATH:
                    # Volume Visualization
                    gv = generated_volume.detach().cpu().numpy()
                    pred_file_name = os.path.join(cfg.DIR.OUT_PATH, imgs_path, rendering_file_path.split('.')[0]+'.obj')
                    utils.voxel.voxel2obj(pred_file_name, gv[0, 1] > cfg.TEST.DEMO_VOXEL_THRESH)
    else:
        raise Exception("error input path")


                


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please Check python version")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
