#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm


def load_id_to_index_map(file_name, key_index=0, value_index=1):
    """
    load item_id->item_index from given file
    :param file_name: the file to store item_id|item_index
    :param key_index: the index to select key
    :param value_index: the index to select value
    :return: dict{key->value}
    """
    assert os.path.exists(file_name), "need to provide non-empty reference file"
    key_to_value = {}
    with open(file_name, "r", encoding="utf-8") as fp:
        print("header = " + fp.readline().strip())
        for p in tqdm(fp, "load info from {} line-by-line...".format(file_name)):
            ps = p.strip().split("|")
            key = ps[key_index]
            value = ps[value_index]
            key_to_value[key] = value
    return key_to_value


def seq_to_str(items, sep="#"):
    """
    merge list of items with sep
    :param items: a list of items
    :param sep: the concatenate notation
    :return: the concatenated string
    """
    return sep.join(map(str, items))


def get_or_create(dictionary, key, default):
    """
    return value if key exists, otherwise create a new value with default
    :param dictionary: the reference dictionary
    :param key: the target key
    :param default: the default value to create a new key-value pair
    :return: the related value
    """
    if key not in dictionary:
        dictionary[key] = default
    return dictionary[key]
