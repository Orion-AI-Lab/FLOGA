from typing import Optional, Union, List

import torch
import torch.nn as nn
from einops import rearrange
from . import initialization as init
from .heads import SegmentationHead

from .decoder import UnetDecoder


class BAM_CD(torch.nn.Module):
    '''
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**

    Returns:
        ``torch.nn.Module``: Unet
    '''
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 2,
        fusion_mode: str = 'conc',
        activation: Optional[Union[str, callable]] = None,
        siamese: bool = True,
        return_features: Optional[bool] = False,
    ):
        super().__init__()

        self.siamese = siamese
        self.return_features = return_features

        self.fusion_mode = fusion_mode

        if self.siamese:
            self.encoder = init.get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )
        else:
            self.encoder1 = init.get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )
            self.encoder2 = init.get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        if self.fusion_mode == 'conc':
            decoder_channels = [i * 2 for i in decoder_channels]
            if self.siamese:
                encoder_channels = [i * 2 for i in self.encoder.out_channels]
            else:
                encoder_channels = [i * 2 for i in self.encoder1.out_channels]
        else:
            if self.siamese:
                encoder_channels = self.encoder.out_channels
            else:
                encoder_channels = self.encoder1.out_channels

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.name = "u-{}".format(encoder_name)

        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)


    def check_input_shape(self, x):
        h, w = x.shape[-2:]

        if self.siamese:
            output_stride = self.encoder.output_stride
        else:
            output_stride = self.encoder1.output_stride

        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def check_fusion_mode(self):
        if self.fusion_mode not in ['conc', 'diff']:
            raise RuntimeError(
                f"Unknown fusion mode: {self.fusion_mode}. Must be one of ['conc', 'diff']."
            )

    def forward(self, x_pre, x_post):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x_pre)
        self.check_input_shape(x_post)

        self.check_fusion_mode()

        # Encode images
        if self.siamese:
            features1 = self.encoder(x_pre)
            features2 = self.encoder(x_post)
        else:
            features1 = self.encoder1(x_pre)
            features2 = self.encoder2(x_post)

        if self.fusion_mode == 'conc':
            features = [torch.cat((x1, x2), dim=1) for x1, x2 in zip(features1, features2)]
        else:
            features = [x2 - x1 for x1, x2 in zip(features1, features2)]

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.return_features:
            return masks, decoder_output
        else:
            return masks
