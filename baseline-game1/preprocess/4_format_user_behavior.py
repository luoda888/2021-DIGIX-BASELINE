#!/user/bin/python
# -*- coding: utf-8 -*-
"""
    Format user-behaviors with user_id_to_index and extract session-based features
    Input: 4_user_behavior.csv device_id|day|pages|music_ids|actions|channel
           device_to_index.csv device_id|device_index
           music_to_index.csv music_id|music_index
    Output: user_behaviors_with_static_info.csv
            device_index|day_id|first_channel|num_actions|action_dist|page_dist|song_dist
"""
import argparse

from tqdm import tqdm
import os
from collections import defaultdict

from utils import load_id_to_index_map


def get_dist_from_list(item_to_list, converter=None):
    """
    Count items by itemId and obtain the itemId to cnt dictionary
    :param item_to_list: the list of itemId
    :param converter: the function to convert raw itemId
    :param default: the default value to fill with NULL
    :return: the dictionary about itemId and count
    """
    num_items = len(item_to_list)
    dist_map = defaultdict(int)
    id_list = []
    for item in item_to_list:
        if converter:
            index = converter.get(item, -1)
        else:
            index = int(item) if item.isdigit() else -1
        dist_map[index] += 1
        id_list.append(index)
    return num_items, dist_map, id_list


def dist_to_str(item_dist):
    """
    Convert item_dist to string
    :param item_dist: the dictionary for item distribution
    :return: the concatenated string
    """
    return "#".join(["{}:{}".format(index, cnt) for (index, cnt) in item_dist.items()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("load user behaviors from dataset to format and extract features...")
    parser.add_argument("--behaviors-file-name", type=str, default="dataset/2021_1_data/4_user_behavior.csv")
    parser.add_argument("--user-index-file-name", type=str, default="dataset/2021_1_data/device_to_index.csv")
    parser.add_argument("--song-index-file-name", type=str, default="dataset/2021_1_data/music_to_index.csv")
    parser.add_argument("--behavior-feature-file-name", type=str,
                        default="dataset/2021_1_data/user_behaviors_with_static_info.csv")
    args = parser.parse_args()

    file_name = args.behaviors_file_name
    assert os.path.exists(file_name), "[ERROR] must provide the behaviors_info file"
    song_to_index_file_name = args.song_index_file_name
    assert os.path.exists(song_to_index_file_name), "[ERROR] must provide the music_id_to_index file"
    user_to_index_file_name = args.user_index_file_name
    assert os.path.exists(user_to_index_file_name), "[ERROR] must provide the device_id_to_index file"
    result_name = args.behavior_feature_file_name

    song_to_index_map = load_id_to_index_map(song_to_index_file_name)
    user_to_index_map = load_id_to_index_map(user_to_index_file_name)
    result_title = "|".join(["device_id", "day_id", "first_channel", "num_actions",
                             "action_dist", "page_dist", "song_dist"])
    with open(file_name, "r") as fp, open(result_name, "w") as gp:
        print("[INFO] raw title = {}".format(fp.readline().strip()))
        gp.write(result_title + "\n")
        for line in tqdm(fp, "load behaviors from {} line-by-line...".format(file_name)):
            items = line.strip().split("|")
            if items[0] not in user_to_index_map or not items[1].isdigit():
                print("\n[WARN] fail to transform {}".format(line.strip()))
                continue
            device_index = user_to_index_map[items[0]]
            day_id = int(items[1])
            _, page_dist, _ = get_dist_from_list(items[2].split("#"))
            num_songs, song_dist, song_list = get_dist_from_list(items[3].split("#"), converter=song_to_index_map)
            num_actions, action_dist, action_list = get_dist_from_list(items[4].split("#"))
            channel_id = int(items[5]) if items[5].isdigit() else -1
            items1 = [device_index, day_id, channel_id, num_actions, dist_to_str(action_dist),
                      dist_to_str(page_dist), dist_to_str(song_dist)]
            line1 = "|".join(map(str, items1))
            gp.write(line1 + "\n")
