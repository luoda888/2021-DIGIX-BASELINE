# -*- coding: utf-8 -*-
import os
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import DataLoader

__all__ = ['OCR_TEST_DataLoader', 'get_test_dataset']


class OCR_TEST_DataLoader(data.Dataset):
    def __init__(self, cfg):
        super(OCR_TEST_DataLoader, self).__init__()
        self.cfg = cfg
        self.long_size = self.cfg.DATASET.TEST.LONG_SIZE
        self.img_paths = self._list_files(cfg.DATASET.TEST.ROOT_PATH)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        ori_name = os.path.split(img_path)[-1]
        img_name = os.path.splitext(os.path.split(img_path)[-1])[0]

        ori_image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        if ori_image.shape[-1] == 4:
            ori_image = ori_image[:, :, :3].copy()

        h, w = ori_image.shape[:2]
        
        if w > h:
            target_w = int(self.long_size)
            scale = self.long_size * 1.0 / w
            target_h = int(h * scale)
            target_h = int(target_h + (32 - target_h % 32))
        else:
            target_h = int(self.long_size)
            scale = self.long_size * 1.0 / h
            target_w = int(w * scale)
            target_w = int(target_w + (32 - target_w % 32))
        if len(ori_image) == 2:
            print(img_name)
        ori_image_bgr = ori_image[:, :, [2, 1, 0]].copy()
        image = np.zeros(shape=(target_h, target_w, 3), dtype=np.uint8)
        _image = cv2.resize(ori_image, dsize=None, fx=scale, fy=scale)
        image[:_image.shape[0], :_image.shape[1], :] = _image

        image = Image.fromarray(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        return ori_image_bgr, image, img_name, scale, ori_name

    def _list_files(self, root_path):
        print('*****TEST DATASET*****:')
        img_paths = []

        for (dirpath, dirnames, filenames) in os.walk(root_path):
            for filename in filenames:
                _filename, ext = os.path.splitext(filename)
                if ext.lower() in self.cfg.DATASET.TEST.IMG_FORMATS:
                    _img_path = os.path.join(dirpath, filename)
                    img_paths.append(_img_path)

        print('***sum_samples: {}'.format(len(img_paths)))
        return img_paths


def get_test_dataset(cfg):
    ocr_dataset = OCR_TEST_DataLoader(cfg)

    dataset_loader = DataLoader(
        ocr_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,)
    return dataset_loader
