# -*- coding: utf-8 -*-

import paddle


class Merger(paddle.nn.Layer):
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = paddle.nn.Sequential(
            paddle.nn.Conv3D(9, 16, kernel_size=3, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(16),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = paddle.nn.Sequential(
            paddle.nn.Conv3D(16, 8, kernel_size=3, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(8),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = paddle.nn.Sequential(
            paddle.nn.Conv3D(8, 4, kernel_size=3, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(4),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = paddle.nn.Sequential(
            paddle.nn.Conv3D(4, 2, kernel_size=3, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(2),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = paddle.nn.Sequential(
            paddle.nn.Conv3D(2, 1, kernel_size=3, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(1),
            paddle.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

    def forward(self, raw_features, coarse_volumes):
        n_views_rendering = coarse_volumes.shape[1]
        raw_features = paddle.split(raw_features, num_or_sections=raw_features.shape[1], axis=1)
        volume_weights = []

        for i in range(n_views_rendering):
            raw_feature = paddle.squeeze(raw_features[i], axis=1)
            # print("torch.Size([batch_size, 9, 32, 32, 32]) ---",raw_feature.shape)

            volume_weight = self.layer1(raw_feature)
            # print("torch.Size([batch_size, 16, 32, 32, 32]) ---",volume_weight.shape)     # 
            volume_weight = self.layer2(volume_weight)
            # print("torch.Size([batch_size, 8, 32, 32, 32]) ---",volume_weight.shape)     # 
            volume_weight = self.layer3(volume_weight)
            # print("torch.Size([batch_size, 4, 32, 32, 32]) ---",volume_weight.shape)     # 
            volume_weight = self.layer4(volume_weight)
            # print("torch.Size([batch_size, 2, 32, 32, 32]) ---",volume_weight.shape)     # 
            volume_weight = self.layer5(volume_weight)
            # print("torch.Size([batch_size, 1, 32, 32, 32]) ---",volume_weight.shape)     # 

            volume_weight = paddle.squeeze(volume_weight, axis=1)
            # print("torch.Size([batch_size, 32, 32, 32]) ---",volume_weight.shape)     # 
            volume_weights.append(volume_weight)

        volume_weights = paddle.transpose(paddle.stack(volume_weights), perm=[1,0,2,3,4])
        volume_weights = paddle.nn.functional.softmax(volume_weights, axis=1)
        # print("torch.Size([batch_size, n_views, 32, 32, 32]) ---",volume_weights.shape)        # 
        # print("torch.Size([batch_size, n_views, 32, 32, 32]) ---",coarse_volumes.shape)        # 
        coarse_volumes = coarse_volumes * volume_weights
        coarse_volumes = paddle.sum(coarse_volumes, axis=1)

        return paddle.clip(coarse_volumes, min=0, max=1)

if __name__ == "__main__":
    from easydict import EasyDict as edict
    __C = edict()
    cfg = __C
    __C.NETWORK = edict()
    __C.NETWORK.LEAKY_VALUE = .2
    
    model = paddle.Model(Merger(cfg))
    model.summary([(4, 2, 9, 32, 32, 32), (4, 2, 32, 32, 32)])