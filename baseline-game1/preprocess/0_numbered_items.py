#!/usr/bin/python
# -*-coding utf-8-*-
"""
   number the device_id and music_id in the information dataset 2_user_info.csv and 3_music_info.csv
   Input:  2_user_info.csv: device_id|gender|age|device|city|is_vip|topics
           3_music_info.csv: music_id|title|artist_id|album|is_paid|comment|comment_cnt
   Output: user_to_index.csv: device_id|device_index
           song_to_index.csv: music_id|music_index
"""

import argparse
from tqdm import tqdm


def number_ids_line_by_line(source_file_name, target_file_name, target_title="device_id|device_index"):
    """
    numbered ids(at first column) line by line from source-file and restore into target-file
    :param source_file_name: information file with id
    :param target_file_name: numbered target file
    :param target_title: the title for target file
    :return: the number of distinct ids
    """
    coded_items = set([])
    numbered_items = 0
    with open(source_file_name, "r", encoding="utf-8") as fp, open(target_file_name, "w") as gp:
        title = fp.readline().strip()
        print("[INFO] deal with title {}".format(title))
        gp.write(target_title + "\n")
        for line in tqdm(fp, "[INFO] loading item-info from {} line-by-line...".format(source_file_name)):
            item_id = line.strip().split("|")[0]
            if not item_id.isdigit():
                print("\n[ERROR] fail to coded with {}".format(line.strip()))
                continue
            if item_id not in coded_items:
                coded_items.add(item_id)
                gp.write("{}|{}\n".format(item_id, numbered_items))
                numbered_items += 1
            else:
                print("\n[WARN] item_id {} is duplicated".format(item_id))
    return numbered_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser("load basic information from dataset and number it...")
    parser.add_argument("--user-info-file-name", type=str, default="dataset/2021_1_data/2_user_info.csv")
    parser.add_argument("--song-info-file-name", type=str, default="dataset/2021_1_data/3_music_info.csv")
    parser.add_argument("--user-numbered-file-name", type=str, default="dataset/2021_1_data/device_to_index.csv")
    parser.add_argument("--song-numbered-file-name", type=str, default="dataset/2021_1_data/music_to_index.csv")
    args = parser.parse_args()

    num_users = number_ids_line_by_line(
        args.user_info_file_name, args.user_numbered_file_name, "device_id|device_index")
    num_songs = number_ids_line_by_line(
        args.song_info_file_name, args.song_numbered_file_name, "music_id|music_index")
    print("[INFO] there are {} distinct users and {} distinct songs".format(num_users, num_songs))
