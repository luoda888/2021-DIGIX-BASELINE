# -*- coding: utf-8 -*-
import sys
import os
import json
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from .dataset_catalog import OCR_DatasetCatalog
from .utils import *
import cv2
import Polygon as plg
import pyclipper
from torch.utils.data import DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['OCR_DataLoader', 'get_train_dataset']


class OCR_DataLoader(data.Dataset):

    def __init__(self, cfg):
        super(OCR_DataLoader, self).__init__()
        self.cfg = cfg
        self.check_all_imgs = cfg.DATASET.TRAIN.CHECK_ALL_IMGS
        self.ocr_dataset_catalog = OCR_DatasetCatalog(cfg.DATASET.TRAIN.ROOT_PATH)
        self.img_paths, self.labels = self._list_files(self.check_all_imgs)
        self.random_resize = RandomResize(cfg)
        self.data_augmentation = Compose(cfg)

    def __len__(self):
        return len(self.img_paths)

    def _cvt2HeatmapImg(self, img):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img

    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]
            label = self.labels[idx]
            img_name = os.path.split(img_path)[-1]

            img_np, bboxes, tags = self._parse_gt(img_path, label) # img_np format: rgb
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                img_np = img_np[:, :, :3].copy()

            image, _ = self.random_resize(img_np)
            h, w = image.shape[:2]

            # get absolute bboxes
            abs_bboxes = []
            for bbox in bboxes:
                _bbox = bbox * np.array([w, h], dtype=np.float32)
                abs_bboxes.append(_bbox.astype(np.int32))

            # get gt_text and training_mask
            gt_text = np.zeros((h, w), dtype=np.float32)
            training_mask = np.ones_like(gt_text, dtype=np.float32)
            for bbox, tag in zip(abs_bboxes, tags):
                cv2.drawContours(gt_text, [bbox], 0, 1, -1)
                if not tag:
                    cv2.drawContours(training_mask, [bbox], 0, 0, -1)

            # get gt_kernels
            gt_kernels = []
            kernel_num = self.cfg.MODEL.HEAD.NUM_CLASS
            shrink_min_scale = self.cfg.MODEL.HEAD.MIN_SCALE
            for i in range(1, kernel_num):
                rate = 1.0 - (1.0 - shrink_min_scale) / (kernel_num - 1.0) * i
                gt_kernel = np.zeros_like(gt_text, dtype=np.float32)
                shrinked_bboxes_i = self._shrink(abs_bboxes, rate)
                for shrinked_bbox in shrinked_bboxes_i:
                    shrinked_bbox = np.array(shrinked_bbox, dtype=np.int32)
                    cv2.drawContours(gt_kernel, [shrinked_bbox], 0, 1, -1)
                gt_kernels.append(gt_kernel)

            # data augmentation
            target = []
            target.append(gt_text)
            target.extend(gt_kernels)
            target.append(training_mask)

            image, target = self.data_augmentation(image, target)

            image = Image.fromarray(image)
            image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

            text_mask = torch.tensor(target[0: kernel_num]).ge(0.5).float()
            training_mask = torch.from_numpy(target[-1]).ge(0.5).float()
            return image, text_mask, training_mask

        except Exception as e:
            print(e)
            print(img_path, label)
            raise Exception('training-dataset __getitem__ error.')

    def _list_files(self, check_all_imgs=False):
        print('*****TRAIN DATASET*****:')
        if check_all_imgs:
            print('check all images.')
        img_paths = []
        labels = []
        for dataset_idx, dataset_name in enumerate(self.cfg.DATASET.TRAIN.DATASET_NAME_LIST):
            # dataset
            print('{dataset_idx}. {dataset_name}: '.format(dataset_idx=dataset_idx + 1,
                                                           dataset_name=dataset_name), end='')
            sys.stdout.flush()

            dataset_attrs = self.ocr_dataset_catalog.get(dataset_name)
            image_path = os.path.join(self.ocr_dataset_catalog.root_path,
                                     dataset_attrs['root_path'])

            with open(os.path.join(self.ocr_dataset_catalog.root_path, dataset_attrs['gt_path']), 'r') as f:
                gt = f.read()
            labels_dict = json.loads(gt)
            for image_name in labels_dict:
                img_paths.append(os.path.join(image_path, image_name))
                labels.append(labels_dict[image_name])

        print('***sum_samples: {}'.format(len(img_paths)))
        return img_paths, labels

    def _parse_gt(self, img_path, label):
        '''
        parse from gt to get normalized 2d bboxes and tags which indicating validity.
        '''
        bboxes = []
        tags = []

        img_pil = Image.open(img_path)
        img_np = np.asarray(img_pil)
        img_h, img_w = img_np.shape[0: 2]

        _img_name, _ = os.path.splitext(os.path.split(img_path)[-1])

        for line in label:
            text = line['label']
            tag = False if text == '' or text[0] == '#' else True
            box = np.array(line['points']).reshape(-1, 2) / np.array([img_w, img_h], dtype=np.float32)
            bboxes.append(box)
            tags.append(tag)

        return img_np, bboxes, tags

    def _perimeter(self, bbox):
        peri = 0.0
        point_num = bbox.shape[0]
        for i in range(point_num):
            _dist = np.sqrt(np.sum((bbox[i] - bbox[(i + 1) % point_num]) ** 2))
            peri += _dist
        return peri

    def _shrink(self, bboxes, rate, max_shr=20):
        rate = rate * rate
        shrinked_bboxes = []
        for bbox in bboxes:
            if bbox.shape[0] < 3:
                continue
            area = plg.Polygon(bbox).area()
            peri = self._perimeter(bbox)

            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = int(area * (1 - rate) / (peri + 1e-3) + 0.5)
            offset = min(offset, max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(np.array(shrinked_bbox)[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)

        return shrinked_bboxes


def get_train_dataset(cfg):
    ocr_dataset = OCR_DataLoader(cfg)
    dataset_loader = DataLoader(
        ocr_dataset,
        batch_size=cfg.DATASET.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.TRAIN.NUM_WORKERS,
        drop_last=cfg.DATASET.TRAIN.DROP_LAST,
        pin_memory=cfg.DATASET.TRAIN.PIN_MEMORY)
    return dataset_loader

