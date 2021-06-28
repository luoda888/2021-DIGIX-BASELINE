# -*- coding=utf-8 -*-
import argparse
import os
import json
import cv2
import sys

import numpy as np

from itertools import groupby
from enum import Enum

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

from recognizer.models.crnn_model import crnn_model_based_on_densenet_crnn_time_softmax_activate
from recognizer.tools.config import config
from recognizer.tools.utils import get_chinese_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ModelType(Enum):
    DENSENET_CRNN_TIME_SOFTMAX = 0


def load_model(model_type, weight):
    if model_type == ModelType.DENSENET_CRNN_TIME_SOFTMAX:
        base_model, _ = crnn_model_based_on_densenet_crnn_time_softmax_activate()
        base_model.load_weights(weight)
    else:
        raise ValueError('parameter model_type error.')
    return base_model


def predict_by_image_path(image_path, input_shape, base_model):
    """
    :param image_path:  input image path
    :param input_shape: input shape
    :param base_model:  base model
    :return:
    """
    input_height, input_width, input_channel = input_shape
    if input_channel == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    scale = image.shape[0] * 1.0 / input_height
    image_width = int(image.shape[1] // scale)
    image = cv2.resize(image, (image_width, input_height))
    image_height, image_width = image.shape[0:2]
    if image_width <= input_width:
        new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
        new_image[:] = 255
        if input_channel == 1:
            image = np.expand_dims(image, axis=2)
        new_image[:, :image_width, :] = image
        image = new_image
    else:
        image = cv2.resize(image, (input_width, input_height))
    text_image = np.array(image, 'f') / 127.5 - 1.0
    text_image = np.reshape(text_image, [1, input_height, input_width, input_channel])
    y_pred = base_model.predict(text_image)
    y_pred = y_pred[:, :, :]
    char_list = list()
    pred_text = list(y_pred.argmax(axis=2)[0])
    for index in groupby(pred_text):
        if index[0] != config.num_class - 1:
            char_list.append(character_map_table[str(index[0])])
    return u''.join(char_list)


def predict(image, input_shape, base_model):
    input_height, input_width, input_channel = input_shape
    scale = image.shape[0] * 1.0 / input_height
    image_width = int(image.shape[1] // scale)
    if image_width <= 0:
        return ''
    image = cv2.resize(image, (image_width, input_height))
    image_height, image_width = image.shape[0:2]
    if image_width <= input_width:
        new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
        new_image[:] = 255
        if input_channel == 1:
            image = np.expand_dims(image, axis=2)
        new_image[:, :image_width, :] = image
        image = new_image
    else:
        image = cv2.resize(image, (input_width, input_height))
    text_image = np.array(image, 'f') / 127.5 - 1.0
    text_image = np.reshape(text_image, [1, input_height, input_width, input_channel])
    y_pred = base_model.predict(text_image)
    y_pred = y_pred[:, :, :]
    char_list = list()
    pred_text = list(y_pred.argmax(axis=2)[0])
    for index in groupby(pred_text):
        if index[0] != config.num_class - 1:
            char_list.append(character_map_table[str(index[0])])
    return u''.join(char_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--char_path', type=str, default='recognizer/tools/dictionary/chars.txt')
    parser.add_argument('--model_path', type=str,
                        default='/path/to/checkpoints/recognizer/weights_crnn-010-28.863.h5')
    parser.add_argument('--null_json_path', type=str,
                        default='/path/to/output/test_null.json')
    parser.add_argument('--test_image_path', type=str,
                        default='/path/to/output/detector_test_output/menu')
    parser.add_argument('--submission_path', type=str,
                        default='/path/to/output/test_submission.json')
    opt = parser.parse_args()

    character_map_table = get_chinese_dict(opt.char_path)
    input_shape = (32, 280, 3)
    model = load_model(ModelType.DENSENET_CRNN_TIME_SOFTMAX, opt.model_path)
    print('load model done.')

    test_label_json_file = opt.null_json_path
    test_image_root_path = opt.test_image_path
    with open(test_label_json_file, 'r', encoding='utf-8') as in_file:
        label_info_dict = json.load(in_file)
        for idx, info in enumerate(label_info_dict.items()):
            image_name, text_info_list = info
            src_image = cv2.imread(os.path.join(test_image_root_path, image_name.split('.')[0] + '.JPEG'))
            # print(os.path.join(test_image_root_path, image_name))
            print('process: {:3d}/{:3d}. image: {}'.format(idx + 1, len(label_info_dict.items()), image_name))
            for index, text_info in enumerate(text_info_list):
                src_point_list = text_info['points']
                sorted_point_list_by_x = sorted(src_point_list, key=lambda x: x[0])
                sorted_point_list_by_y = sorted(src_point_list, key=lambda x: x[1])
                # print(src_point_list)
                # print(sorted_point_list_by_x, sorted_point_list_by_y)
                # print(src_image)
                crop_image = src_image[sorted_point_list_by_y[0][1]:sorted_point_list_by_y[-1][1],
                             sorted_point_list_by_x[0][0]:sorted_point_list_by_x[-1][0], :]
                rec_result = predict(crop_image, input_shape, model)
                text_info['label'] = rec_result

    save_label_json_file = opt.submission_path
    with open(save_label_json_file, 'w') as out_file:
        out_file.write(json.dumps(label_info_dict))


'''bash
python recognizer/predict.py
'''
