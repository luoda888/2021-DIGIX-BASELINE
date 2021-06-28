#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Format the device active logs and divide into two parts (1~30 and 31~60)
    Input: 1_device_active.csv: device_id|active_days
    Output: active_logs_to_train.csv: device_index|active_flags|persist_labels
            active_logs_to_predict.csv: device_index|active_flags
"""

import argparse
import os
from tqdm import tqdm
from collections import defaultdict
from utils import load_id_to_index_map, seq_to_str

TIME_TO_SPLIT_USAGE = 30


def active_or_not(t, usage):
    """
    judge whether user is active or not at day-t
    :param t: the specific day-id
    :param usage: the active logs
    :return: 1 or 0, indicate active or not
    """
    return 1 if t in usage else 0


def determine_persist_flags(usage):
    """
    determine the persist labels: whether to be active at day (31, 32, 33, 37, 44, 60)
    :param usage: the active logs
    :return: a list of persist flags, in which the element is 0/1 indicating active or not.
    """
    return [active_or_not(t1, usage) for t1 in [31, 32, 33, 37, 44, 60]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("load device active logs, format and divide them...")
    parser.add_argument("--active-log-file-name", type=str, default="dataset/2021_1_data/1_device_active.csv")
    parser.add_argument("--train-log-file-name", type=str, default="dataset/2021_1_data/active_logs_to_train.csv")
    parser.add_argument("--predict-log-file-name", type=str, default="dataset/2021_1_data/active_logs_to_predict.csv")
    parser.add_argument("--user-index-file-name", type=str, default="dataset/2021_1_data/device_to_index.csv")
    args = parser.parse_args()

    source_filename = args.active_log_file_name
    assert os.path.exists(source_filename), "[ERROR] need to provide non-empty active-log-file"
    train_filename = args.train_log_file_name
    predict_filename = args.predict_log_file_name
    device_to_index_map = load_id_to_index_map(args.user_index_file_name)

    usage_dict = defaultdict(set)
    with open(source_filename, "r") as fp:
        print("[INFO] raw_title = " + fp.readline().strip())
        for line in tqdm(fp, "[INFO] load active logs from {} line-by-line...".format(args.active_log_file_name)):
            items = line.strip().split("|")
            device_id = items[0]
            active_days = [int(x) for x in items[1].split("#")]
            if device_id not in device_to_index_map:
                print("\n[INFO] process with illegal device_code {}".format(device_id))
                continue
            usage_dict[device_to_index_map[device_id]].update(active_days)
    num_users = len(usage_dict)
    with open(train_filename, "w") as gp1, open(predict_filename, "w") as gp2:
        gp1.write("device_index|active_flags|persist_labels\n")
        gp2.write("device_index|active_flags\n")
        for device_index, behaviors in tqdm(usage_dict.items(), "[INFO] re-formatted usage into sequence"):
            seq1 = [active_or_not(day_id + 1, behaviors) for day_id in range(TIME_TO_SPLIT_USAGE)]
            seq2 = [active_or_not(day_id + 31, behaviors) for day_id in range(TIME_TO_SPLIT_USAGE)]
            persist_flags = determine_persist_flags(behaviors)
            gp1.write("{}|{}|{}\n".format(device_index, seq_to_str(seq1, "#"), seq_to_str(persist_flags, "#")))
            gp2.write("{}|{}\n".format(device_index, seq_to_str(seq2, "#")))
    print("[INFO] convert device_active to device_usage_seq for {} distinct users".format(num_users))
