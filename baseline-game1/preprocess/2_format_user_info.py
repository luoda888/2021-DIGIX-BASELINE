#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Extract categorical features from user_info
    Input: 2_user_info.csv: device_id|gender|age|device|city|is_vip|topics
    Output: user_with_side_info:
            device_index|gender_index|age_index|phone_index|city_index|vip_index
"""

import os
import argparse
from tqdm import tqdm
from utils import load_id_to_index_map, get_or_create

DEFAULT_AGE_INDEX = 5
DEFAULT_GENDER_INDEX = 2
DEFAULT_VIP_FLAG_INDEX = 2


def write_index_to_file(fp, id_to_index):
    """
    write id_to_index dictionary into fp
    :param fp: the file descriptor
    :param id_to_index: the dict about id->index
    :return: the number of id-index pairs
    """
    for id, index in tqdm(id_to_index.items(), "write id and index to file..."):
        fp.write("{}|{}\n".format(id, index))
    return len(id_to_index)


def optional_transform(value, default=-1):
    """
    Optional transform value
    1. value is digit => int(value)
    2. value is NULL or other => default
    :param value: data to transform
    :param default: otherwise to return
    :return: an Integer
    """
    if value.isdigit():
        return int(value)
    return default


if __name__ == '__main__':
    parser = argparse.ArgumentParser("load device information, format and extract categorical features...")
    parser.add_argument("--device-info-file-name", type=str, default="dataset/2021_1_data/2_user_info.csv")
    parser.add_argument("--user-index-file-name", type=str, default="dataset/2021_1_data/device_to_index.csv")
    parser.add_argument("--user-side-info-file-name", type=str, default="dataset/2021_1_data/user_with_side_info.csv")
    parser.add_argument("--phone-index-file-name", type=str, default="dataset/2021_1_data/phone_id_to_index.csv")
    parser.add_argument("--city-index-file-name", type=str, default="dataset/2021_1_data/city_id_to_index.csv")
    args = parser.parse_args()

    source_filename = args.device_info_file_name
    assert os.path.exists(source_filename), "must provide the user_info file"
    result_filename = args.user_side_info_file_name
    phone_id_to_index = args.phone_index_file_name
    city_id_to_index = args.city_index_file_name

    device_to_index_map = load_id_to_index_map(args.user_index_file_name)

    with open(source_filename, "r", encoding="utf-8") as fp, open(result_filename, "w") as gp, \
            open(phone_id_to_index, "w") as index1, open(city_id_to_index, "w") as index2:
        source_title = fp.readline().strip()
        print("[INFO] iterate data as {}".format(source_title))
        gp.write("device_index|gender_index|age_index|phone_index|city_index|vip_index\n")
        index1.write("phone_id|phone_index\n")
        index2.write("city_id|city_index\n")
        index_phone_map = {}
        index_city_map = {}
        for line in tqdm(fp, "[INFO] iterate information of users line-by-line..."):
            items = line.strip().split("|")
            device_id = items[0]
            if device_id not in device_to_index_map:
                print("\n[WARN] process with illegal device_code {}".format(device_id))
                continue
            device_index = device_to_index_map[device_id]
            gender_cate = optional_transform(items[1], DEFAULT_GENDER_INDEX)
            age_cate = optional_transform(items[2], DEFAULT_AGE_INDEX)
            phone_no = get_or_create(index_phone_map, items[3], len(index_phone_map))
            city_no = get_or_create(index_city_map, items[4], len(index_phone_map))
            vip_type = optional_transform(items[5], DEFAULT_VIP_FLAG_INDEX)
            gp.write("{}|{}|{}|{}|{}|{}\n".format(device_index, gender_cate, age_cate, phone_no, city_no, vip_type))
        print("[INFO] there are {} distinct phone_id".format(write_index_to_file(index1, index_phone_map)))
        print("[INFO] there are {} distinct phone_id".format(write_index_to_file(index2, index_city_map)))
