import os
import cv2
import math
import argparse
import operator
import collections
import sys
import torch
import json

import numpy as np

from tqdm import tqdm
from functools import reduce

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

from detector.config import cfg
from detector.models import OCR_DETECTOR
from detector.dataset import get_test_dataset
from detector.postprocess import simple_dilate


decoder_methods = {
    'SIMPLE_DILATE': simple_dilate
}

tt = {}


def predict(model, data_loader, dst_path, decoder_method, output_box):
    for batch_data in tqdm(data_loader):

        # get data
        ori_image, image, image_name, scale, ori_name = batch_data

        tt[ori_name[0]] = []
        # print(image_name)
        ori_image = np.array(ori_image[0], dtype=np.uint8)
        image_name = image_name[0]
        scale = 1.0 / scale[0].numpy() * (4.0 / model.get_scale())

        # model inference
        if os.environ['CUDA_VISIBLE_DEVICES'] is not None:
            image = image.cuda()

        score, kernels = model(image)
        score = score.cpu().detach().numpy()
        kernels = kernels.cpu().detach().numpy()
        # # decoder
        polygons = decoder_method(score, kernels)
        # polygons = decoder_method(kernels, score)

        _Polyghons = []
        for polygon in polygons:
            _polygon = polygon.reshape(-1, 2) * scale
            _Polyghons.append(_polygon)

        # save result
        if not os.path.isdir(os.path.join(dst_path, 'menu/')):
            os.makedirs(os.path.join(dst_path, 'menu/'))

        if output_box:
            _Min_area_box = []
            for polygon in _Polyghons:
                t = {}
                _min_area_box = cv2.minAreaRect(polygon.astype(np.float32))
                _min_area_box = np.array(cv2.boxPoints(_min_area_box), dtype=np.int32)
                cv2.drawContours(ori_image, [_min_area_box], 0, (0, 255, 0), 2)
                # print(_min_area_box)
                _min_area_box[_min_area_box < 0] = 0
                bb = _min_area_box.tolist()
                t["label"] = ""
                t["points"] = bb
                # print(bb)
                tt[ori_name[0]].append(t)
                _Min_area_box.append(_min_area_box)
            write_result_as_txt(_Min_area_box, os.path.join(dst_path, 'menu/{}.txt'.format(image_name)))
        else:
            for polygon in _Polyghons:
                cv2.drawContours(ori_image, [polygon.astype(np.int32)], 0, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(dst_path, 'menu/{}.JPEG'.format(image_name)), ori_image)


def sort_to_clockwise(points):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    clockwise_points = sorted(points, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    return clockwise_points


def write_result_as_txt(bboxes, dst_path):
    dir_path = os.path.split(dst_path)[0]
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    lines = []
    for bbox in bboxes:
        bbox = bbox.reshape(-1, 2)
        bbox = np.array(list(sort_to_clockwise(bbox)))[[3, 0, 1, 2]].copy().reshape(-1)
        values = [int(v) for v in bbox]
        line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
        lines.append(line)
    with open(dst_path, 'w') as f:
        for line in lines:
            f.write(line)


if __name__ == '__main__':
    # get config
    parser = argparse.ArgumentParser('Hyperparams')
    parser.add_argument('--config', type=str, default=r'detector/config/resnet50.yaml', help='config yaml file.')
    parser.add_argument('--cuda', type=str, default="0")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    cfg.merge_from_file(args.config)
    cfg.freeze()

    # model
    print('model info:')
    print('=> 1.backbone: ', cfg.MODEL.BACKBONE.ARCH)
    print('=> 2.neck: ', cfg.MODEL.NECK.ARCH)
    print('=> 3.head: ', cfg.MODEL.HEAD.ARCH)
    model = OCR_DETECTOR(cfg=cfg)
    if os.environ['CUDA_VISIBLE_DEVICES'] is not None:
        model = model.cuda()

    print("loading trained model '{}'".format(cfg.MODEL.TEST.CKPT_PATH))
    ckpt = torch.load(cfg.MODEL.TEST.CKPT_PATH)
    state_dict = ckpt if 'state_dict' not in ckpt.keys() else ckpt['state_dict']
    model_state_dict = collections.OrderedDict()

    for key, value in state_dict.items():
        if key.startswith('module'):
            _key = '.'.join(key.split('.')[1:])
        else:
            _key = key
        model_state_dict[_key] = value
    model.load_state_dict(model_state_dict)
    print('load trained parameters successfully.')
    model.eval()

    data_loader = get_test_dataset(cfg)
    with torch.no_grad():
        predict(model, data_loader,
                cfg.MODEL.TEST.RES_PATH,
                decoder_methods[cfg.MODEL.TEST.DECODER_METHOD],
                cfg.MODEL.TEST.OUTPUT_BOX)

    with open(os.path.join(cfg.MODEL.TEST.NULL_JSON_PATH, "test_null.json"), "w")as f:
        qq = json.dumps(tt)
        f.write(qq)


'''bash
python detector/predict.py
'''

