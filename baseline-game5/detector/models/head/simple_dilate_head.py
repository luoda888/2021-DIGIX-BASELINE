import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SIMPLE_DILATE_HEAD']

class SIMPLE_DILATE_HEAD(nn.Module):

    def __init__(self, cfg):
        self.is_train = False # default mode: inference
        self.scale = cfg.MODEL.HEAD.SCALE
        self.interpolation_mode = cfg.MODEL.HEAD.INTERPOLATION_MODE
        self.align_corners = cfg.MODEL.HEAD.ALIGN_CORNERS
        self.use_bias = cfg.MODEL.HEAD.USE_BIAS
        super(SIMPLE_DILATE_HEAD, self).__init__()
        self.conv1 = nn.Conv2d(cfg.MODEL.HEAD.IN_CHANNEL,
                               cfg.MODEL.HEAD.MID_CHANNEL,
                               kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        self.bn1 = nn.BatchNorm2d(cfg.MODEL.HEAD.MID_CHANNEL)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cfg.MODEL.HEAD.MID_CHANNEL,
                               cfg.MODEL.HEAD.NUM_CLASS,
                               kernel_size=1, stride=1, padding=0)

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature):
        out = self.relu1(self.bn1(self.conv1(feature)))
        out = self.conv2(out)
        _, _, h, w = out.shape
        out = F.interpolate(out,
                            size=(int(self.scale * h), int(self.scale * w)),
                            mode=self.interpolation_mode,
                            align_corners=self.align_corners)
        if self.is_train:
            return out

        text = out[0, 0, :, :]
        kernel = out[0, 3, :, :]
        score = torch.sigmoid(text)
        kernel = ((text > 1.0) & (kernel > 1.0)).type(torch.uint8)
        return (score, kernel)