# -*- coding=utf-8 -*-
import argparse
import os
import json

import cv2

global_image_num = 0
char_set = set()


def extract_train_data(src_image_root_path, src_label_json_file, save_image_path, save_txt_path):
    global global_image_num, char_set
    with open(src_label_json_file, 'r', encoding='utf-8') as in_file:
        label_info_dict = json.load(in_file)
        with open(os.path.join(save_txt_path, 'train.txt'), 'a', encoding='utf-8') as out_file:
            for image_name, text_info_list in label_info_dict.items():
                src_image = cv2.imread(os.path.join(src_image_root_path, image_name))
                for text_info in text_info_list:
                    try:
                        text = text_info['label']
                        for char in text:
                            char_set.add(char)
                        src_point_list = text_info['points']
                        sorted_point_list_by_x = sorted(src_point_list, key=lambda x: x[0])
                        sorted_point_list_by_y = sorted(src_point_list, key=lambda x: x[1])
                        crop_image = src_image[sorted_point_list_by_y[0][1]:sorted_point_list_by_y[-1][1],
                                     sorted_point_list_by_x[0][0]:sorted_point_list_by_x[-1][0], :]
                        if crop_image.size == 0:
                            continue
                        crop_image_name = '{}.jpg'.format(global_image_num)
                        global_image_num += 1
                        cv2.imwrite(os.path.join(save_image_path, crop_image_name), crop_image)
                        out_file.write('{}\t{}\n'.format(crop_image_name, text))
                    except:
                        pass

        for image_name, text_info_list in label_info_dict.items():
            for text_info in text_info_list:
                text = text_info['label']
                text = text.replace('\r', '').replace('\n', '')
                for char in text:
                    char_set.add(char)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_train_image_path', type=str,
                        default='/path/to/tmp_data/recognizer_images')
    parser.add_argument('--save_train_txt_path', type=str,
                        default='/path/to/tmp_data/recognizer_txts')
    parser.add_argument('--train_image_common_root_path', type=str,
                        default='/path/to/official_data/train_image_common')
    parser.add_argument('--common_label_json_file', type=str,
                        default='/path/to/official_data/train_label_common.json')

    parser.add_argument('--train_image_special_root_path', type=str,
                        default='/path/to/official_data/train_image_special')
    parser.add_argument('--special_label_json_file', type=str,
                        default='/path/to/official_data/train_label_special.json')

    opt = parser.parse_args()

    save_train_image_path = opt.save_train_image_path
    save_train_txt_path = opt.save_train_txt_path

    train_image_common_root_path = opt.train_image_common_root_path
    common_label_json_file = opt.common_label_json_file
    extract_train_data(train_image_common_root_path,
                       common_label_json_file,
                       save_train_image_path,
                       save_train_txt_path)

    train_image_special_root_path = opt.train_image_special_root_path
    special_label_json_file = opt.special_label_json_file

    extract_train_data(train_image_special_root_path,
                       special_label_json_file,
                       save_train_image_path,
                       save_train_txt_path)

    print('Image num is {}.'.format(global_image_num))

    char_list = list(char_set)
    char_list.sort()

    with open('chars.txt', 'a', encoding='utf-8') as out_file:
        for char in char_list:
            out_file.write('{}\n'.format(char))

'''bash
python recognizer/tools/extract_train_data.py
'''

