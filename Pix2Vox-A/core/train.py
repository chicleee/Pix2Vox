# -*- coding: utf-8 -*-
#

import os
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
from models.refiner import Refiner

def train_net(cfg):
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = paddle.io.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    #num_workers=0  , # cfg.TRAIN.NUM_WORKER>0时报错，因为dev/shm/太小  https://blog.csdn.net/ctypyb2002/article/details/107914643
                                                    #pin_memory=True,
                                                    use_shared_memory=False,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = paddle.io.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  #num_workers=1,
                                                  #pin_memory=True,
                                                  shuffle=False)

    # Set up networks # paddle.Model prepare fit save
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    merger = Merger(cfg)
    refiner = Refiner(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))

    # # Initialize weights of networks # paddle的参数化不同，参见API
    # encoder.apply(utils.network_utils.init_weights)
    # decoder.apply(utils.network_utils.init_weights)
    # merger.apply(utils.network_utils.init_weights)

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA, verbose=True)
    decoder_lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=cfg.TRAIN.DECODER_LEARNING_RATE,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA, verbose=True)
    merger_lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=cfg.TRAIN.MERGER_LEARNING_RATE,
                                                               milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                               gamma=cfg.TRAIN.GAMMA, verbose=True)
    refiner_lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=cfg.TRAIN.REFINER_LEARNING_RATE,
                                                                milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA, verbose=True)
    # Set up solver
    # if cfg.TRAIN.POLICY == 'adam':
    encoder_solver = paddle.optimizer.Adam(learning_rate=encoder_lr_scheduler,parameters=encoder.parameters())
    decoder_solver = paddle.optimizer.Adam(learning_rate=decoder_lr_scheduler,parameters=decoder.parameters())
    merger_solver = paddle.optimizer.Adam(learning_rate=merger_lr_scheduler, parameters=merger.parameters())
    refiner_solver = paddle.optimizer.Adam(learning_rate=refiner_lr_scheduler,parameters=refiner.parameters())

    # if torch.cuda.is_available():
    #     encoder = torch.nn.DataParallel(encoder).cuda()
    #     decoder = torch.nn.DataParallel(decoder).cuda()
    #     merger = torch.nn.DataParallel(merger).cuda()

    # Set up loss functions
    bce_loss = paddle.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        # load
        encoder_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "encoder.pdparams"))
        encoder_solver_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "encoder_solver.pdopt"))
        encoder.set_state_dict(encoder_state_dict)
        encoder_solver.set_state_dict(encoder_solver_state_dict)
        decoder_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "decoder.pdparams"))
        decoder_solver_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "decoder_solver.pdopt"))
        decoder.set_state_dict(decoder_state_dict)
        decoder_solver.set_state_dict(decoder_solver_state_dict)

        if cfg.NETWORK.USE_MERGER:
            merger_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "merger.pdparams"))
            merger_solver_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "merger_solver.pdopt"))
            merger.set_state_dict(merger_state_dict)
            merger_solver.set_state_dict(merger_solver_state_dict)
        
        if cfg.NETWORK.USE_REFINER:
            refiner_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "refiner.pdparams"))
            refiner_solver_state_dict = paddle.load(os.path.join(cfg.CONST.WEIGHTS, "refiner_solver.pdopt"))
            refiner.set_state_dict(refiner_state_dict)
            refiner_solver.set_state_dict(refiner_solver_state_dict)

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
              (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    # train_writer = SummaryWriter()
    # val_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    train_writer=LogWriter(os.path.join(log_dir, 'train'))
    val_writer=LogWriter(os.path.join(log_dir, 'val'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()

        # # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        # print("****debug: length of train data loder",n_batches)
        for batch_idx, (rendering_images, ground_truth_volumes) in enumerate(train_data_loader()):
            # # debug
            # if batch_idx>1:
            #     break

            # Measure data time
            data_time.update(time() - batch_end_time)
            # print("****debug: batch_idx",batch_idx)
            # print(rendering_images.shape)
            # print(ground_truth_volumes.shape)
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)

            # Train the encoder, decoder, and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volumes = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            # else:
            #     mergered_volumes = paddle.mean(generated_volumes, aixs=1)

            encoder_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refiner(generated_volumes)
                refiner_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
            # else:
            #     refiner_loss = encoder_loss

             # Gradient decent
            encoder_solver.clear_grad ()
            decoder_solver.clear_grad ()
            merger_solver.clear_grad ()
            refiner_solver.clear_grad()

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
            # else:
            #     encoder_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            merger_solver.step()
            refiner_solver.step()

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.numpy())
            refiner_losses.update(refiner_loss.numpy())

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar(tag='EncoderDecoder/BatchLoss', step=n_itr, value=encoder_loss.numpy())
            train_writer.add_scalar('Refiner/BatchLoss', value = refiner_loss.numpy(), step = n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if (batch_idx % int(cfg.CONST.INFO_BATCH )) == 0:
                print('[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
                    (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                    data_time.val, encoder_loss.numpy(), refiner_loss.numpy()))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar(tag='EncoderDecoder/EpochLoss', step=epoch_idx + 1, value=encoder_losses.avg)
        train_writer.add_scalar('Refiner/EpochLoss', value = refiner_losses.avg, step = epoch_idx + 1)

        # update scheduler each step
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        merger_lr_scheduler.step()
        refiner_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, encoder_losses.avg, refiner_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, merger, refiner)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d' % (epoch_idx + 1)),
                                                 epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver,
                                                 merger, merger_solver, refiner, refiner_solver, best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-ckpt'), epoch_idx + 1, encoder,
                                                 encoder_solver, decoder, decoder_solver, merger, merger_solver, refiner, refiner_solver,
                                                 best_iou, best_epoch)

    # # Close SummaryWriter for TensorBoard
    # train_writer.close()
    # val_writer.close()
