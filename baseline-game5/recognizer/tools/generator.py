# -*- coding=utf-8 -*-


import os
import random
import traceback
import time

import cv2
import numpy as np

from recognizer.tools.config import config


class BatchIndices:
    def __init__(self, total_num, batch_size, is_training=True):
        self.total_num = total_num
        self.batch_size = batch_size
        self.is_training = is_training
        self.index = None
        self.curr = None
        self.is_epoch_end = False
        self.reset()

    def reset(self):
        self.index = np.random.permutation(self.total_num) if self.is_training else np.arange(0, self.total_num)
        self.curr = 0

    def __next__(self):
        self.is_epoch_end = False
        if self.curr >= self.total_num:
            self.reset()
            self.is_epoch_end = True
        remaining_next = min(self.batch_size, self.total_num - self.curr)
        res = self.index[self.curr:self.curr + remaining_next]
        self.curr += remaining_next
        return res, self.is_epoch_end


class Generator:
    def __init__(self, root_path, input_map_file, batch_size, max_label_length, input_shape, is_training):
        self.root_path = root_path
        self.input_map_file = input_map_file
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.input_shape = input_shape
        self.is_training = is_training
        self.epoch_time = 0
        self.image_to_label = self.parse_map_file()
        self.batch_indexes = BatchIndices(len(self.image_to_label), self.batch_size, self.is_training)

    def parse_map_file(self):
        res = list()
        with open(self.input_map_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for i in lines:
                res.append(i.strip().replace('\n', '').replace('\r', ''))
        dic = dict()
        for i in res:
            path, values = i.split('\t')
            dic[path] = values.split(' ')
        return dic

    def num_samples(self):
        return len(self.image_to_label)

    def __next__(self):
        images_name = [image_name for image_name, image_label in self.image_to_label.items()]
        image_name_array = np.array(images_name)
        input_height, input_width, input_channel = self.input_shape
        sequence_length = 280
        start = 0
        while True:
            if config.is_debug:
                start = time.time()
            batch_index, is_epoch_end = next(self.batch_indexes)
            curr_bath_size = len(batch_index)
            try:
                batch_image_name_array = image_name_array[batch_index]
                label = np.ones([curr_bath_size, self.max_label_length]) * 10000
                input_length = np.zeros([curr_bath_size, 1])
                label_length = np.zeros([curr_bath_size, 1])
                input_images = np.zeros((curr_bath_size, input_height, input_width, input_channel), dtype=np.float)
                index = 0
                for image_name in batch_image_name_array:
                    try:
                        if input_channel == 1:
                            image = cv2.imread(os.path.join(self.root_path, image_name), cv2.IMREAD_GRAYSCALE)
                        else:
                            image = cv2.imread(os.path.join(self.root_path, image_name), cv2.IMREAD_COLOR)

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
                            if input_channel == 1:
                                image = np.expand_dims(image, axis=2)
                        image = np.array(image, 'f') / 127.5 - 1.0
                    except Exception as e:
                        print('skipped image {}. exception: {}'.format(image_name, e))
                        continue
                    input_images[index] = image
                    label_length[index] = len(self.image_to_label[image_name])
                    input_length[index] = sequence_length
                    label[index, :len(self.image_to_label[image_name])] = [int(k) for k in
                                                                           self.image_to_label[image_name]]
                    index += 1
                label = np.delete(label, [i for i in range(index, curr_bath_size)], axis=0)
                input_length = np.delete(input_length, [i for i in range(index, curr_bath_size)], axis=0)
                label_length = np.delete(label_length, [i for i in range(index, curr_bath_size)], axis=0)
                input_images = np.delete(input_images, [i for i in range(index, curr_bath_size)], axis=0)

                inputs = {'input_data': input_images,
                          'label': label,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros([index])}
                self.epoch_time += (time.time() - start)
                if config.is_debug and is_epoch_end:
                    print("\nThe current total time for epoch to load data is {0}.".format(self.epoch_time))
                    self.epoch_time = 0
                del image, label, input_images, label_length, input_length
                yield inputs, outputs
            except Exception as e:
                print('{0} is wrong, error is {1}. {2}'.format(image_name_array[batch_index], str(e),
                                                               traceback.format_exc()))
                self.__next__()
