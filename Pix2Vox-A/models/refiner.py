# -*- coding: utf-8 -*-

import paddle

class Refiner(paddle.nn.Layer):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = paddle.nn.Sequential(
            paddle.nn.Conv3D(1, 32, kernel_size=4, padding=2, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(32),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            paddle.nn.MaxPool3D(kernel_size=2, stride=2, padding=0)
        )
        self.layer2 = paddle.nn.Sequential(
            paddle.nn.Conv3D(32, 64, kernel_size=4, padding=2, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(64),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            paddle.nn.MaxPool3D(kernel_size=2, stride=2, padding=0)
        )
        self.layer3 = paddle.nn.Sequential(
            paddle.nn.Conv3D(64, 128, kernel_size=4, padding=2, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(128),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            paddle.nn.MaxPool3D(kernel_size=2, stride=2, padding=0)
        )
        self.layer4 = paddle.nn.Sequential(
            paddle.nn.Linear(8192, 2048),
            paddle.nn.ReLU()
        )
        self.layer5 = paddle.nn.Sequential(
            paddle.nn.Linear(2048, 8192),
            paddle.nn.ReLU()
        )
        self.layer6 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(128, 64, kernel_size=4, stride=2, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=False),
            paddle.nn.BatchNorm3D(64),
            paddle.nn.ReLU()
        )
        self.layer7 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(64, 32, kernel_size=4, stride=2, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=False),
            paddle.nn.BatchNorm3D(32),
            paddle.nn.ReLU()
        )
        self.layer8 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(32, 1, kernel_size=4, stride=2, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=False),
            paddle.nn.Sigmoid()
        )

    def forward(self, coarse_volumes):
        # print(coarse_volumes.shape)

        volumes_32_l = paddle.reshape(coarse_volumes, [-1, 1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX])
        # print("paddle.Size([batch_size, 1, 32, 32, 32])---->", volumes_32_l.shape)

        volumes_16_l = self.layer1(volumes_32_l)
        # print("paddle.Size([batch_size, 32, 16, 16, 16])---->", volumes_16_l.shape)

        volumes_8_l = self.layer2(volumes_16_l)
        # print("paddle.Size([batch_size, 64, 8, 8, 8])---->", volumes_8_l.shape)
 
        volumes_4_l = self.layer3(volumes_8_l)
        # print("paddle.Size([batch_size, 128, 4, 4, 4])---->", volumes_4_l.shape)
 
        flatten_features = self.layer4(paddle.reshape(volumes_4_l, [-1, 8192]))
        # print("paddle.Size([batch_size, 2048])---->", flatten_features.shape)

        flatten_features = self.layer5(flatten_features)
        # print("# paddle.Size([batch_size, 8192])---->", flatten_features.shape)

        volumes_4_r = volumes_4_l + paddle.reshape(flatten_features, [-1, 128, 4, 4, 4])
        # print("paddle.Size([batch_size, 128, 4, 4, 4])---->", volumes_4_r.shape)
 
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print("paddle.Size([batch_size, 64, 8, 8, 8])---->", volumes_8_r.shape)

        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print("paddle.Size([batch_size, 32, 16, 16, 16])---->", volumes_16_l.shape)

        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print("paddle.Size([batch_size, 1, 32, 32, 32])---->", volumes_32_r.shape)

        return paddle.reshape(volumes_32_r, [-1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX])


if __name__ == "__main__":
    from easydict import EasyDict as edict
    __C = edict()
    cfg = __C
    cfg.CONST = edict()
    cfg.CONST.N_VOX=32
    cfg.NETWORK = edict()
    cfg.NETWORK.LEAKY_VALUE = .2
    model = paddle.Model(Refiner(cfg))
    model.summary((4, 32, 32, 32))