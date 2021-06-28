import torch
import cv2
import random
import numpy as np

__all__ = ['Compose', 'RandomResize', 'RandomHorizontalFlip', 'RandomRotate', 'RandomCrop', 'Resize']


class Compose(object):

    def __init__(self, cfg, is_layout=False):
        super(Compose, self).__init__()
        data_augmentation = {
            'random_resize': RandomResize(cfg),
            'random_horizontal_flip': RandomHorizontalFlip(),
            'random_rotate': RandomRotate(cfg),
            'random_crop': RandomCrop(cfg),
        }
        if is_layout:
            self.transforms = [data_augmentation[op] for op in cfg.DATASET.LAYOUT.TRAIN.DATA_AUGMENTATION]
        else:
            self.transforms = [data_augmentation[op] for op in cfg.DATASET.TRAIN.DATA_AUGMENTATION]

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):

    def __init__(self, cfg):
        super(RandomResize, self).__init__()
        self.random_scales = cfg.DATASET.AUGMENTATION.RANDOM_RESIZE.RANDOM_SCALES
        self.min_size = cfg.DATASET.AUGMENTATION.RANDOM_RESIZE.MIN_SIZE

    def __call__(self, image, target=None):
        '''
        random scale from ndarray format Image
        '''

        h, w = image.shape[0: 2]
        scale = 1
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)

        w, h = w * scale, h * scale
        random_scale = random.sample(self.random_scales, 1)[0]
        if min(w, h) * random_scale <= self.min_size:
            random_scale = (self.min_size + 10) * 1.0 / min(w, h)
        w, h = int(w * random_scale), int(h * random_scale)
        image = cv2.resize(image, dsize=(w, h))

        if isinstance(target, list):
            target = [cv2.resize(t, dsize=(w, h)) for t in target]
        elif target is None:
            return image, scale * random_scale
        else:
            target = cv2.resize(target, dsize=(w, h))
        return image, target, scale * random_scale


class Resize(object):

    def __init__(self, long_size):
        super(Resize, self).__init__()
        self.long_size = long_size

    def __call__(self, image, target=None):
        '''
        random scale from ndarray format Image
        '''
        h, w = image.shape[0: 2]
        if h > w:
            h_target = self.long_size
            scale = self.long_size * 1.0 / h
            w_target = scale * w
            w_target += (32 - w_target % 32)
            w_target = int(w_target)
        else:
            w_target = self.long_size
            scale = self.long_size * 1.0 / w
            h_target = scale * h
            h_target += (32 - h_target % 32)
            h_target = int(h_target)

        image = cv2.resize(image, dsize=(w_target, h_target))
        image_t = np.zeros(shape =(self.long_size, self.long_size, 3), dtype=np.uint8)
        image_t[:h_target, :w_target, :] = image

        return image_t, np.array([w_target, h_target], dtype=np.float32)


class RandomHorizontalFlip(object):

    def __init__(self):
        pass

    def __call__(self, image, target=None):
        if random.random() < 0.5:
            image = np.flip(image, axis=1)

            if target is None:
                return image
            elif isinstance(target, list):
                target = [np.flip(t, axis=1) for t in target]
            else:
                target = np.flip(target, axis=1)
            return image, target

        else:
            if target is None:
                return image
            else:
                return image, target


class RandomRotate(object):

    def __init__(self, cfg):
        super(RandomRotate, self).__init__()
        self.max_angle = cfg.DATASET.AUGMENTATION.RANDOM_ROTATE.MAX_ANGLE
        self.base_angles = cfg.DATASET.AUGMENTATION.RANDOM_ROTATE.BASE_ANGLES

    def __call__(self, image, target):
        base_angle = random.choice(self.base_angles)
        angle = base_angle + (random.random() * 2 - 1) * self.max_angle
        h, w = image.shape[0: 2]
        rotate_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, rotate_matrix, (w, h))
        if target is None:
            return image
        elif isinstance(target, list):
            _tmp_img = target[0]
            _tmp_h, _tmp_w = _tmp_img.shape[0: 2]
            rotate_matrix = cv2.getRotationMatrix2D((_tmp_w / 2.0, _tmp_h / 2.0), angle, 1)
            target = [cv2.warpAffine(t, rotate_matrix, (_tmp_w, _tmp_h)) for t in target]
        else:
            _tmp_h, _tmp_w = target.shape[0: 2]
            rotate_matrix = cv2.getRotationMatrix2D((_tmp_w / 2.0, _tmp_h / 2.0), angle, 1)
            target = cv2.warpAffine(target, rotate_matrix, (_tmp_w, _tmp_h))
        return image, target


class RandomCrop(object):

    def __init__(self, cfg):
        super(RandomCrop, self).__init__()
        self.min_size = cfg.DATASET.AUGMENTATION.RANDOM_CROP.MIN_SIZE

    def __call__(self, image, target=None):

        h, w = image.shape[0: 2]
        th, tw = self.min_size, self.min_size
        if h == th and w == tw:
            if target is None:
                return image
            else:
                return image, target

        gt_text = target[0] if isinstance(target, list) else target
        scale = gt_text.shape[0] * 1.0 / image.shape[0]
        h, w = h * scale, w * scale
        th, tw = th * scale, tw * scale
        if random.random() > 3.0 / 8.0 and np.max(gt_text) > 0:
            tl = np.min(np.where(gt_text > 0), axis=1) - np.array([th, tw])
            tl[tl < 0] = 0

            br = np.max(np.where(gt_text > 0), axis=1) - np.array([th, tw])
            br[br < 0] = 0

            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = random.randint(tl[0], br[0])
            j = random.randint(tl[1], br[1])
        else:
            i = random.randint(0, int(h - th))
            j = random.randint(0, int(w - tw))

        image = image[int(i / scale): int((i + th) / scale),
                      int(j / scale): int((j + tw) / scale), :].copy()

        if isinstance(target, list):
            target = [t[i: int(i + th), j: int(j + tw)].copy() for t in target]
        else:
            target = target[i: int(i + th), j: int(j + tw)].copy()
        return image, target





