#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Generate baseline features
    Input:
        1. user_with_side_information.csv: Indicate the categorical features of users
        2. song_with_side_information.csv: Indicate the categorical features of songs
        3. user_behaviors_with_static_info.csv: Indicate the static features for users' daily behaviors
        4. active_logs_to_train.csv: Indicate the active flags among day-1 to day-30
    Output:
        1. train_features_and_labels.csv: features and labels for training
        2. predict_features.csv: features for predicting
        3. missing_behaviors.csv: logs which in user_behaviors_with_static_info.csv but not in user_active_seq.csv
"""

import os
import argparse
from tqdm import tqdm


def is_day_important(target_day_id):
    """
    judge whether to involve in this dataset
    :param target_day_id: the day_id to judge
    :return: true or false, indicate the result of judgement
    """
    return target_day_id < 31


def load_features_from_file(filename, func, result):
    """
    load features line-by-line from the given file (i.e., named filename) and convert line via func to result
    :param filename: the specific given file
    :param func: the convert function
    :param result: to restore converted result
    """
    if not os.path.exists(filename):
        print("[ERROR] fail to convert features with an empty file {}".format(filename))
        return
    with open(filename, "r", encoding="utf-8") as fp:
        print("[INFO] title = {}".format(fp.readline().strip()))
        for line in tqdm(fp, "[INFO] load information from {} line-by-line...".format(filename)):
            items = line.strip().split("|")
            func(items, result)


def update_coder(coder, name):
    """
    update coder by given key
    :param coder: a triple item about [dict, num, name]
    :param name: a given key
    """
    if name not in coder[0]:
        coder[0][name] = coder[1]
        coder[1] += 1


def save_label_coders(coders, data_dir):
    """
    restore item to item_index relationships into data_dir and coder_name
    :param coders: a list of item to item_index relationships
    :param data_dir: the directory to save relationships
    """
    assert os.path.exists(data_dir), "[ERROR] must provide an exists template directory"
    for coder in coders:
        coder_map = coder[0]
        coder_size = coder[1]
        coder_name = coder[2]
        file_name = os.path.join(data_dir, coder_name)
        with open(file_name, "w") as fp:
            print("[INFO] write {} coded relationship to {}".format(coder_size, coder_name))
            fp.write("id|index\n")
            for raw_id, coded_id in coder_map.items():
                fp.write("{}|{}\n".format(raw_id, coded_id))


def get_device_static_features(user_static_features_file_name):
    """
    get users' static features from given file
    :param user_static_features_file_name: the specific filename about users' static features
    :return: a dictionary about device_index to encoded static features
    """
    def deal_with_device_side_info(items, result_map):
        """
        load users' side information from file
        :param items: items in str format split by "|"
        :param result_map: a dictionary to store device_index to side-information
        """
        # device_index|gender_index|age_index|phone_index|city_index|vip_index
        key = int(items[0])
        value = [int(x) for x in items[1:]]
        result_map[key] = value
    assert os.path.exists(user_static_features_file_name), "[ERROR] must provide the user_with_side_info.csv"
    # output: device_index -> [gender_id, age_id, is_vip, city_no, phone_id]
    table_user_side_info = {}
    load_features_from_file(user_static_features_file_name, deal_with_device_side_info, table_user_side_info)
    return table_user_side_info


def parse_item_distribution(dist_str):
    """
    parse the distribution of items from given str
    :param dist_str: a str about "value1:count1#...#valueN:countN"
    :return: a dictionary about value to count
    """
    dist_str_list = dist_str.split("#")
    dist_map = {}
    for p in dist_str_list:
        ps = p.split(":")
        val = int(ps[0])
        cnt = int(ps[1])
        dist_map[val] = cnt
    return dist_map


def get_device_behaviors(items, table):
    """
    load behavior-based features from items split by "|"
    :param items: items split by "|", device_id|day_id|first_channel|num_actions|action_dist|page_dist|song_dist
    :param table: a dictionary to restore device_index to behavior-based features
    """
    # device_index|day_id|first_channel|num_actions|action_dist|page_dist|song_dist
    device_index = int(items[0])
    day_id = int(items[1])
    first_channel = int(items[2])
    num_interactions = int(items[3])
    action_dist = parse_item_distribution(items[4])
    page_dist = parse_item_distribution(items[5])
    song_dist = parse_item_distribution(items[6])
    if not is_day_important(day_id):
        return
    if device_index not in table:
        table[device_index] = {}
    table[device_index][day_id] = [first_channel, num_interactions, action_dist, page_dist, song_dist]


def update_coder_via_dist(coder, dist):
    """
    update coder based the given distribution
    :param coder: a triple item about [dict, num, name]
    :param dist: a given distribution
    :return: an updated distribution
    """
    raw_items = dist.items()
    new_dist = {}
    for k, v in raw_items:
        if k not in coder[0]:
            coder[0][k] = coder[1]
            coder[1] += 1
        new_dist[coder[0][k]] = v
    return new_dist


def get_usage_info_with_styles(user_behavior_log_file_name, song_with_style_file_name):
    """
    merge song_with_side_info and user_behavior_logs to get the behavior-based static features
    :param user_behavior_log_file_name: the specific filename about users' behavior logs
    :param song_with_style_file_name: the specific filename about songs' static features
    :return: a dictionary restored device_index to behavior-related features for everyday
                1. channel_distribution;
                2. action_distribution;
                3. page_distribution;
                4. song_distribution;
                5. styles_distribution;
            a list of dictionaries about item-to-index [dict, num_values, name]
    """

    def get_song_and_styles(items, result_map):
        """
        get song_index to styles relationships
        :param items: [music_index, pay_flag, comment_cnt, style_categories]
        :param result_map: a dictionary to store song_index to style_categories
        """
        song_index = int(items[0])
        styles = [style for style in items[3].split("#") if style.isdigit()]
        if len(styles) > 0:
            result_map[song_index] = [int(style) for style in styles]

    # music_index -> [pay_flag, comment_cnt, style_categories]
    assert os.path.exists(song_with_style_file_name), "[ERROR] must provide the song_with_side_information.csv"
    table_song_to_styles = {}
    load_features_from_file(song_with_style_file_name, get_song_and_styles, table_song_to_styles)
    # device_index -> {day_id -> [first_channel, num_interactions, action_dist, page_dist, song_dist, styles_dist]}
    assert os.path.exists(user_behavior_log_file_name), "[ERROR] must provide the user_behaviors_with_static_info.csv"
    table_device_behaviors = {}
    load_features_from_file(user_behavior_log_file_name, get_device_behaviors, table_device_behaviors)
    coders = [
        [{-1: 0}, 1, name] for name in
        ["channel_to_index.csv", "action_to_index.csv", "page_to_index.csv", "style_to_index.csv"]
    ]
    for device_index, behavior_features in tqdm(table_device_behaviors.items(), "[INFO] match styles via song-id..."):
        for behavior_day, static_features in behavior_features.items():
            song_dist = static_features[4]
            style_dist = {}
            for song, cnt in song_dist.items():
                if song not in table_song_to_styles:
                    continue
                for style in table_song_to_styles[song]:
                    style_dist[style] = style_dist.get(style, 0) + cnt
            table_device_behaviors[device_index][behavior_day] = static_features + [style_dist]
    for device_index, behavior_features in tqdm(table_device_behaviors.items(), "[INFO] update coder for category..."):
        for behavior_day, behavior_features in behavior_features.items():
            update_coder(coders[0], behavior_features[0])
            behavior_features[0] = coders[0][0][behavior_features[0]]
            behavior_features[2] = update_coder_via_dist(coders[1], behavior_features[2])
            behavior_features[3] = update_coder_via_dist(coders[2], behavior_features[3])
            behavior_features[5] = update_coder_via_dist(coders[3], behavior_features[5])
            table_device_behaviors[device_index][behavior_day] = behavior_features
    return table_device_behaviors, coders


def get_active_sequence(items, table):
    """
    parase the active day ids from items split by "|"
    :param items: items split by "|"
    :param table: the relationships
    """
    device_index = int(items[0])
    status = [int(x) for x in items[1].split("#")]
    labels = [] if len(items) <= 2 else [int(x) for x in items[2].split("#")]
    table[device_index] = (status, labels)


def get_top_list(dist, top_k):
    """
    learn the top-k frequency category id from distribution
    :param dist: the given distribution
    :param top_k: the number of k
    :return: a list of top-k item-id
    """
    if len(dist) == 0:
        return [0, 0, 0]
    items = list(dist.items())
    items.sort(key=lambda kv: -kv[1])
    num_items = len(items)
    result = [items[idx][0] for idx in range(min(top_k, num_items))] + [0] * max(0, top_k - num_items)
    return result


def dist_to_str(dist_map):
    """
    concatenate the distributions into string
    :param dist_map: a distribution map as key->value
    :return: a concatenated string
    """
    return "#".join(["{}:{}".format(k, v) for (k, v) in dist_map.items()])


def get_interval_features(start_day_id, stop_day_id, usage_features, dist_feature, scale_feature):
    """
    learn the features among different time interval [start_day_id, stop_day_id)
    :param start_day_id: the start day-id of the interval
    :param stop_day_id: the stop day-id of the interval
    :param usage_features: the usage-flags
    :param dist_feature: the distributions of each category to fill
    :param scale_feature: the scale features to fill
    """
    # {day_id -> [first_channel, num_interactions, action_dist, page_dist, song_dist, styles_dist]}
    action_dist = {}
    styles_dist = {}
    pages_dist = {}
    channels_dist = {}
    num_interactions = 0
    song_dist = {}
    for iter_day_id in range(start_day_id, stop_day_id):
        if iter_day_id not in usage_features:
            continue
        usages = usage_features[iter_day_id]
        channels_dist[usages[0]] = channels_dist.get(usages[0], 0) + 1
        num_interactions += usages[1]
        for action_id, cnt in usages[2].items():
            action_dist[action_id] = action_dist.get(action_id, 0) + cnt
        for page_id, cnt in usages[3].items():
            pages_dist[page_id] = pages_dist.get(page_id, 0) + cnt
        for song_id, cnt in usages[4].items():
            song_dist[song_id] = song_dist.get(song_id, 0) + cnt
        for style_id, cnt in usages[5].items():
            styles_dist[style_id] = song_dist.get(style_id, 0) + cnt
    num_songs = sum(cnt for (_, cnt) in song_dist.items())
    distinct_songs = len(song_dist)
    scale_feature.extend([num_interactions, num_songs, distinct_songs])
    dist_feature.extend([dist_to_str(action_dist), dist_to_str(pages_dist), dist_to_str(styles_dist)])


def list_to_str(values, sep=","):
    """
    concatenate item in values with sep
    :param values: a list of items to concatenate
    :param sep: the notation to concatenate
    :return: a concatenated string
    """
    return sep.join(map(str, values))


def generate_features(file_descriptor, table_device_active, static_features, usage_features, t0, t1):
    """
    calculate the specific features for given active-status and users-related features in [t0, t1)
    :param file_descriptor: the file descriptor to write features and labels
    :param table_device_active: the dictionary to store the interacted status of devices
    :param static_features: the static features of users
    :param usage_features: the usage features of users
    :param t0: the start day-id to calculate features
    :param t1: the stop day-id to calculate features
    :return: the list of bad-cases which fail to match behaviors
    """
    bad_case_list = []
    for device_index, (status, labels) in tqdm(table_device_active.items(), "[INFO] iterate users one-by-one..."):
        if device_index not in static_features:
            print("[WARN] fail to find {} from static_features".format(device_index))
            continue
        cate_features = static_features[device_index]
        dist_features = []
        scale_features = []
        for date in range(1, 31):
            if status[date - 1] == 0:
                cate_features.extend([0] * 11)
                scale_features.append(0)
            elif device_index not in usage_features or date not in usage_features[device_index]:
                bad_case_list.append((device_index, date))
                cate_features.extend([0] * 11)
                scale_features.append(0)
            else:
                first_channel = usage_features[device_index][date][0]
                num_actions = usage_features[device_index][date][1]
                top_actions = get_top_list(usage_features[device_index][date][2], 3)
                top_pages = get_top_list(usage_features[device_index][date][3], 3)
                top_styles = get_top_list(usage_features[device_index][date][5], 3)
                cate_features.extend([1, first_channel] + top_actions + top_pages + top_styles)
                scale_features.append(num_actions)
        if device_index not in usage_features:
            scale_features.extend([0, 0, 0] * 5)
            dist_features.extend(["", "", ""] * 5)
        else:
            # last 30 days
            get_interval_features(t1 - 30, t1, usage_features[device_index], dist_features, scale_features)
            # last 14 days
            get_interval_features(t1 - 14, t1, usage_features[device_index], dist_features, scale_features)
            # last 7 days
            get_interval_features(t1 - 7, t1, usage_features[device_index],
                                  dist_features, scale_features)
            # last 3 days
            get_interval_features(t1 - 3, t1, usage_features[device_index], dist_features, scale_features)
            # last 2 days
            get_interval_features(t1 - 2, t1, usage_features[device_index], dist_features, scale_features)
        features_str = "{}|{}|{}|{}".format(device_index, list_to_str(cate_features),
                                            list_to_str(dist_features), list_to_str(scale_features))
        if len(labels) > 0:
            file_descriptor.write("{}|{}\n".format(features_str, list_to_str(labels)))
        else:
            file_descriptor.write("{}\n".format(features_str))
    return bad_case_list


def generate_total_features(train_active_file_name, train_dataset_file_name, predict_active_file_name,
                            predict_dataset_file_name, device_static_features, device_usage_features):
    """
    generate the total features and restore into result_file
    :param train_active_file_name: a given filename to indicate active sequence at 1-30
    :param train_dataset_file_name: a given filename to restore features at 1-30
    :param predict_active_file_name: a given filename to indicate active sequence at 31-60
    :param predict_dataset_file_name: a given filename to restore features at 31-60
    :param device_static_features: a dictionary about users' static features
    :param device_usage_features: a dictionary about users' usage-based features
    :return: a list of bad cases
    """
    assert os.path.exists(train_active_file_name), "[ERROR] must give an existed active file"
    assert os.path.exists(predict_active_file_name), "[ERROR] must give an existed active file"
    # device_id -> (interacted_days, labels)
    table_device_training_status = {}
    load_features_from_file(train_active_file_name, get_active_sequence, table_device_training_status)
    table_device_predicting_status = {}
    load_features_from_file(predict_active_file_name, get_active_sequence, table_device_predicting_status)
    # whole features
    # device_static_features:
    # device_id --> [gender_id, age_id, city_no, phone_code, is_vip]
    # device_usage_features:
    # device_id --> {day_id -> [first_channel, num_interactions, action_dist, page_dist, song_dist, styles_dist]}
    with open(train_dataset_file_name, "w") as fp1, open(predict_dataset_file_name, "w") as fp2:
        fp1.write("device_id|category_features|distribution_features|numeric_features|labels\n")
        fp2.write("device_id|category_features|distribution_features|numeric_features\n")
        bad_cases = generate_features(fp1, table_device_training_status, device_static_features,
                                      device_usage_features, 1, 31)
        bad_cases += generate_features(fp2, table_device_predicting_status, device_static_features,
                                       device_usage_features, 31, 60)
        return bad_cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser("generate features for training and predicting...")
    parser.add_argument("--user-static-features", type=str,
                        default="dataset/2021_1_data/user_with_side_information.csv")
    parser.add_argument("--song-static-features", type=str,
                        default="dataset/2021_1_data/song_with_side_information.csv")
    parser.add_argument("--user-behaviors-features", type=str,
                        default="dataset/2021_1_data/user_behaviors_with_static_info.csv")
    parser.add_argument("--training-active-logs", type=str, default="dataset/2021_1_data/active_logs_to_train.csv")
    parser.add_argument("--training-dataset", type=str, default="dataset/2021_1_data/train_features_and_labels.csv")
    parser.add_argument("--predicting-active-logs", type=str, default="dataset/2021_1_data/active_logs_to_predict.csv")
    parser.add_argument("--predicting-dataset", type=str, default="dataset/2021_1_data/predict_features.csv")
    parser.add_argument("--temp-dir", type=str, default="dataset/2021_1_data")
    parser.add_argument("--bad-case", type=str, default="dataset/2021_1_data/missing_behaviors.csv")
    args = parser.parse_args()

    # device_id --> [gender_id, age_id, is_vip, city_id, phone_code]
    device_to_static_features = get_device_static_features(args.user_static_features)
    # behaviors --> matched styles
    # device_id --> {day_id -> [first_channel, num_interactions, action_dist, page_dist, song_dist, styles_dist]}
    device_usage_with_styles, cate_coders = get_usage_info_with_styles(args.user_behaviors_features,
                                                                       args.song_static_features)
    save_label_coders(cate_coders, args.temp_dir)
    # device_id -> [gender_id, age_id, is_vip, phone_id] + interactions + static
    missing_lst = generate_total_features(args.training_active_logs, args.training_dataset,
                                          args.predicting_active_logs, args.predicting_dataset,
                                          device_to_static_features, device_usage_with_styles)
    print("[WARN] {} bad cases totally in behavior.csv".format(len(missing_lst)))
    with open(args.bad_case, "w") as fp:
        fp.write("device_id|day_id\n")
        for device_index, bad_day_id in missing_lst:
            fp.write("{}|{}\n".format(device_index, bad_day_id))


