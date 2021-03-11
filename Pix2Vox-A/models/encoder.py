# -*- coding: utf-8 -*-

import paddle
from paddle.vision.models import vgg16

class Encoder(paddle.nn.Layer):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = vgg16(pretrained=False, batch_norm=True)
        self.vgg = paddle.nn.Sequential(*list(vgg16_bn.features.children())[:27])
        self.layer1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(512, 512, kernel_size=1, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm2D(512),
            paddle.nn.ELU(),
        )
        self.layer2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(512, 256, kernel_size=3, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm2D(256),
            paddle.nn.ELU(),
            paddle.nn.MaxPool2D(kernel_size=4)
        )
        self.layer3 = paddle.nn.Sequential(
            paddle.nn.Conv2D(256, 128, kernel_size=3, weight_attr=paddle.nn.initializer.KaimingNormal(), bias_attr=paddle.nn.initializer.Constant(value=0.0)),
            paddle.nn.BatchNorm2D(128),
            paddle.nn.ELU()
        )

        # # Don't update params in VGG16
        # for param in vgg16_bn.parameters():
        #     param.requires_grad = False

    def forward(self, rendering_images):
        # print("rendering_images.shape", rendering_images.shape)  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = paddle.transpose(rendering_images, perm=[1, 0, 2, 3, 4]) # pytorch:rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        # print("after transpose shape", rendering_images.shape) # [2, 4, 3, 224, 224]
        rendering_images = paddle.split(rendering_images, num_or_sections=rendering_images.shape[0], axis=0) # return list @@ len() = num_or_sections 跟pytorch区别大
        # print("after split len", len(rendering_images))
        image_features = []

        for img in rendering_images:
            features = self.vgg(paddle.squeeze(img, axis=0))
            # print(features.shape)    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.shape)    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer2(features)
            # print(features.shape)    # torch.Size([batch_size, 256, 6, 6])
            features = self.layer3(features)
            # print(features.shape)    # torch.Size([batch_size, 128, 4, 4])
            image_features.append(features)

        image_features = paddle.stack(image_features)
        # print(image_features.shape)
        image_features = paddle.transpose(image_features, perm=[1, 0, 2, 3, 4])
        # print(image_features.shape)  # torch.Size([batch_size, n_views, 128, 4, 4])
        return image_features

if __name__ == "__main__":
    from easydict import EasyDict as edict
    __C = edict()
    cfg = __C
    model = paddle.Model(Encoder(cfg))
    model.summary((4, 2, 3, 224, 224))