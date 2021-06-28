#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Extract categorical features from music_info
    Input: 3_music_info.csv: music_id|title|artist_id|album|is_paid|comment|comment_cnt
           5_artist_info.csv: artist_id|artist|style
    Output: song_with_side_info.csv
            music_index|pay_flag|comment_cnt|style_categories
"""

import argparse
from tqdm import tqdm
from collections import defaultdict
import os

from utils import load_id_to_index_map, get_or_create, seq_to_str

DEFAULT_PAID_FLAG = 2


def write_index_to_file(filename, id_to_index, title="style_name|style_index"):
    """
    write id_to_index dictionary into fp
    :param filename: the name of the result file
    :param id_to_index: the dict about id->index
    :param title: the title for result
    :return: the number of id-index pairs
    """
    with open(filename, "w", encoding="utf-8") as gp:
        gp.write(title + "\n")
        for id, index in tqdm(id_to_index.items(), "write id and index to file..."):
            gp.write("{}|{}\n".format(id, index))
    return len(id_to_index)


def load_artist_with_styles(filename):
    """
    load artist_id and styles from artist_info.csv
    :param filename: filename of artist_info.csv
    :return: the artist_to_style_ids dictionary and the the style_name_to_id dictionary
    """
    artist_to_styles = defaultdict(set)
    style_name_to_index = {}
    with open(filename, "r", encoding="utf-8") as fp:
        title = fp.readline().strip()
        print("[INFO] title of {} is {}".format(filename, title))
        for line in tqdm(fp, "[INFO] load artist_info line-by-line..."):
            items = line.strip().split("|")
            if not items[0].isdigit():
                print("[WARN] fail to transform {} for illegal artistId".format(line))
                continue
            artist_id = items[0]
            style_names = items[2].split("#")
            for style in style_names:
                style_index = get_or_create(style_name_to_index, style, len(style_name_to_index))
                artist_to_styles[artist_id].add(style_index)
    return artist_to_styles, style_name_to_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser("load music information from dataset and number it...")
    parser.add_argument("--music-info-file-name", type=str, default="dataset/2021_1_data/3_music_info.csv")
    parser.add_argument("--artist-info-file-name", type=str, default="dataset/2021_1_data/5_artist_info.csv")
    parser.add_argument("--song-index-file-name", type=str, default="dataset/2021_1_data/music_to_index.csv")
    parser.add_argument("--song-info-file-name", type=str, default="dataset/2021_1_data/song_with_side_info.csv")
    parser.add_argument("--style-index-file-name", type=str, default="dataset/2021_1_data/style_id_to_index.csv")
    args = parser.parse_args()

    music_filename = args.music_info_file_name
    assert os.path.exists(music_filename), "[ERROR] must provide the music_info file"
    artist_filename = args.artist_info_file_name
    assert os.path.exists(artist_filename), "[ERROR] must provide the artist_filename file"
    music_index_filename = args.song_index_file_name
    assert os.path.exists(music_index_filename), "[ERROR] must provide the music_index_filename file"
    result_filename = args.song_info_file_name
    style_index_filename = args.style_index_file_name

    artist_to_style_indices, style_to_index = load_artist_with_styles(artist_filename)
    print("[INFO] write {} style->index pairs into file".format(
        write_index_to_file(style_index_filename, style_to_index)))

    music_id_to_index = load_id_to_index_map(music_index_filename, 0, 1)
    with open(music_filename, "r", encoding="utf-8") as fp, open(result_filename, "w") as gp:
        print("[INFO] the title of the raw music_info is {}".format(fp.readline().strip()))
        gp.write("music_index|pay_flag|comment_cnt|style_categories\n")
        for line in tqdm(fp, "load raw music-info from file line-by-line..."):
            items = line.strip().split("|")
            if items[0] not in music_id_to_index:
                print("\n[WARN] fail to transform illegal music_id {}".format(items[0]))
                continue
            music_index = music_id_to_index[items[0]]
            styles = artist_to_style_indices.get(items[1], set([]))
            paid_flag = int(items[4]) if items[4].isdigit() else DEFAULT_PAID_FLAG
            comment_cnt = int(items[6]) if items[5].isdigit() else 0
            gp.write("{}|{}|{}|{}\n".format(music_index, paid_flag, comment_cnt, seq_to_str(list(styles))))
