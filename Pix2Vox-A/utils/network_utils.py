# -*- coding: utf-8 -*-

import paddle
import os
from datetime import datetime as dt


def var_or_cuda(x):
    # if torch.cuda.is_available():
    x = x.cuda(blocking=False ) 

    return x


# def init_weights(m):
#     if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
#         torch.nn.init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             torch.nn.init.constant_(m.bias, 0)
#     elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d: #paddle有报错
#         torch.nn.init.constant_(m.weight, 1)
#         torch.nn.init.constant_(m.bias, 0)
#     elif type(m) == torch.nn.Linear:
#         torch.nn.init.normal_(m.weight, 0, 0.01)
#         torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, encoder, encoder_solver, decoder, decoder_solver, merger,
                     merger_solver, refiner, refiner_solver, best_iou, best_epoch):
    print('[INFO] %s Saving %s checkpoint to %s ...' % (dt.now(), epoch_idx, file_path))
    print('[INFO] best_epoch %s best_iou %s ...' % (best_epoch, best_iou.numpy ( ) ))
    paddle.save(encoder.state_dict(), os.path.join(file_path, "encoder.pdparams"))
    paddle.save(encoder_solver.state_dict(), os.path.join(file_path, "encoder_solver.pdopt"))
    paddle.save(decoder.state_dict(), os.path.join(file_path, "decoder.pdparams"))
    paddle.save(decoder_solver.state_dict(), os.path.join(file_path, "decoder_solver.pdopt"))
    paddle.save(refiner.state_dict(), os.path.join(file_path, "refiner.pdparams"))
    paddle.save(refiner_solver.state_dict(), os.path.join(file_path, "refiner_solver.pdopt"))
    if cfg.NETWORK.USE_MERGER:
        paddle.save(merger.state_dict(), os.path.join(file_path, "merger.pdparams"))
        paddle.save(merger_solver.state_dict(), os.path.join(file_path, "merger_solver.pdopt"))
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
