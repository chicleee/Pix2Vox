# -*- coding: utf-8 -*-

import paddle


class Decoder(paddle.nn.Layer):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(256, 128, kernel_size=4, stride=2, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(128),
            paddle.nn.ReLU()
        )
        self.layer2 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(128, 64, kernel_size=4, stride=2, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(64),
            paddle.nn.ReLU()
        )
        self.layer3 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(64, 32, kernel_size=4, stride=2, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(32),
            paddle.nn.ReLU()
        )
        self.layer4 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(32, 8, kernel_size=4, stride=2, padding=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm3D(8),
            paddle.nn.ReLU()
        )
        self.layer5 = paddle.nn.Sequential(
            paddle.nn.Conv3DTranspose(8, 1, kernel_size=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = paddle.transpose(image_features, perm=[1, 0, 2, 3, 4])
        image_features = paddle.split(image_features, num_or_sections = image_features.shape[0], axis=0)
        gen_voxels = []
        raw_features = []

        for features in image_features:
            gen_voxel = paddle.reshape(features, [-1, 256, 2, 2, 2])
            # print("torch.Size([batch_size, 256, 2, 2, 2]) ---", gen_voxel.shape)
            gen_voxel = self.layer1(gen_voxel)
            # print("torch.Size([batch_size, 128, 4, 4, 4]) ---", gen_voxel.shape)   
            gen_voxel = self.layer2(gen_voxel)
            # print("torch.Size([batch_size, 64, 8, 8, 8]) ---", gen_voxel.shape) 
            gen_voxel = self.layer3(gen_voxel)
            # print("torch.Size([batch_size, 32, 16, 16, 16]) ---", gen_voxel.shape) 
            gen_voxel = self.layer4(gen_voxel)
            # print("torch.Size([batch_size, 8, 32, 32, 32]) ---", gen_voxel.shape)     
            raw_feature = gen_voxel
            gen_voxel = self.layer5(gen_voxel)
            # print("torch.Size([batch_size, 1, 32, 32, 32]) ---", gen_voxel.shape)    
            raw_feature = paddle.concat(x = [raw_feature, gen_voxel], axis=1)
            # print("torch.Size([batch_size, 9, 32, 32, 32]) ---",raw_feature.shape) 

            gen_voxels.append(paddle.squeeze(gen_voxel, axis=1))
            raw_features.append(raw_feature)

        gen_voxels = paddle.transpose(paddle.stack(gen_voxels), perm=[1,0,2,3,4])
        raw_features = paddle.transpose(paddle.stack(raw_features), perm=[1,0,2,3,4,5])
        # print("torch.Size([batch_size, n_views, 32, 32, 32]) ---", gen_voxels.shape)
        # print("torch.Size([batch_size, n_views, 9, 32, 32, 32]) ---",raw_features.shape) 
        return raw_features, gen_voxels

if __name__ == "__main__":
    from easydict import EasyDict as edict
    __C = edict()
    cfg = __C
    model = paddle.Model(Decoder(cfg))
    model.summary((4, 2, 128, 4, 4))