import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FPN']

def conv3x3(in_channel, out_channel, kernel_size, padding, use_bias):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=kernel_size, padding=padding, bias=use_bias)

def conv1x1(in_channel, out_channel, kernel_size, padding, use_bias):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=kernel_size, padding=padding, bias=use_bias)

class FPN(nn.Module):

    def __init__(self, cfg):
        self.in_channels = cfg.MODEL.NECK.IN_CHANNELS
        self.out_channels = cfg.MODEL.NECK.OUT_CHANNELS
        self.interpolation_mode = cfg.MODEL.NECK.INTERPOLATION_MODE
        self.align_corners = cfg.MODEL.NECK.ALIGN_CORNERS
        self.use_bias = cfg.MODEL.NECK.USE_BIAS
        super(FPN, self).__init__()

        self.reduce_layers, self.smooth_layers = self._make_layers(self.in_channels, self.out_channels)
        self._init_parameters()

    def _make_layers(self, in_channels, out_channels):
        reduce_layers = []
        smooth_layers = []
        for idx, in_channel in enumerate(in_channels):
            reduce_layer = nn.Sequential(
                conv1x1(in_channel, out_channels[idx], 1, 0, use_bias=self.use_bias),
                nn.BatchNorm2d(out_channels[idx]),
                nn.ReLU(inplace=True))
            reduce_layers.append(reduce_layer)
            if idx != len(in_channels) - 1:
                smooth_layer = nn.Sequential(
                    conv3x3(out_channels[idx], out_channels[idx], 3, 1, use_bias=self.use_bias),
                    nn.BatchNorm2d(out_channels[idx]),
                    nn.ReLU(inplace=True))
                smooth_layers.append(smooth_layer)

        return nn.Sequential(*reduce_layers), nn.Sequential(*smooth_layers)

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample_like(self, x, y):
        _, _, h, w = y.shape
        return F.interpolate(x, size=(h, w),
                             mode=self.interpolation_mode,
                             align_corners=self.align_corners)

    def _upsample_and_add(self, x, y):
        return self._upsample_like(x, y) + y

    def forward(self, features):
        output_features = []
        for i in range(len(features) - 1, -1, -1):
            reduce_feature = self.reduce_layers[i](features[i])
            if i == len(features) - 1:
                output_feature = reduce_feature
                output_features.append(output_feature)
            else:
                output_feature = self._upsample_and_add(output_feature, reduce_feature)
                output_feature = self.smooth_layers[i](output_feature)
                output_features.append(output_feature)

        _, _, h, w = output_features[-1].shape
        for idx, output_feature in enumerate(output_features):
            if idx != len(output_features) - 1:
                output_features[idx] = F.interpolate(output_feature, size=(h, w),
                                                     mode=self.interpolation_mode,
                                                     align_corners=self.align_corners)
            else:
                output_features[idx] = output_feature

        output_features.reverse()
        output = torch.cat(output_features, dim=1)

        return output