import torch.nn as nn
from .backbone import *
from .neck import *
from .head import *
from .loss import *

__all__ = ['OCR_DETECTOR']

BACKBONE = {
    'resnet50': resnet50,
}

NECK = {
    'fpn': FPN,
}

HEAD = {
    'simple_dilate_head': SIMPLE_DILATE_HEAD,
}

LOSS = {
    'dice_loss': DiceLoss,
}

class OCR_DETECTOR(nn.Module):

    def __init__(self, cfg):
        super(OCR_DETECTOR, self).__init__()
        self.backbone = BACKBONE[cfg.MODEL.BACKBONE.ARCH](cfg)
        self.neck = NECK[cfg.MODEL.NECK.ARCH](cfg)
        self.head = HEAD[cfg.MODEL.HEAD.ARCH](cfg)
        self.scale = cfg.MODEL.HEAD.SCALE

    def set_mode(self, mode='TRAIN'):
        if hasattr(self.head, 'is_train'):
            if mode.upper() == 'TRAIN':
                self.head.is_train = True
            elif mode.upper() == 'INFERENCE':
                self.head.is_train = False
            else:
                raise NotImplemented
        else:
            pass

    def get_scale(self):
        return self.scale

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        y = self.head(x)
        return y







