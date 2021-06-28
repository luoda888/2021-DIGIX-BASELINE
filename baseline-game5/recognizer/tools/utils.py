# -*- coding=utf-8 -*-


from bidict import bidict


def get_chinese_dict(input_file_path):
    keys = bidict()
    i = 0
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for char in input_file:
            char = char.replace('\r', '').replace('\n', '')
            if char == '#':
                i += 1
                continue
            keys['{0}'.format(i)] = char
            i += 1
    return keys
