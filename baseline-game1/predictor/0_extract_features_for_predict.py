#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Extract valid features from the whole feature dataset
    Input: total_features_to_predict.csv: the whole features
    Output: predict_features.csv: the valid features to predict
"""

import os
import argparse
from tqdm import tqdm
from Features import CategoryFeature, DistributionFeatures

category_to_num = {
    "gender": 3, "age": 6, "city": 352,
    "phone": 339, "vip": 3, "channels": 48,
    "actions": 9, "pages": 55, "styles": 6
}


def generate_titles():
    """
    generate the title for valid features
    :return: a title in string-format
    """
    label_title = ["label_{}d".format(idx) for idx in [1, 2, 3, 7, 14, 30]]
    feature_title = ["gender_id", "age_id", "city_id", "phone_id", "vip_flag"]
    feature_title += ["channel_id_{}".format(i+1) for i in range(30)]
    feature_title += ["top1_action_id_{}".format(i+1) for i in range(30)]
    feature_title += ["top1_page_id_{}".format(i+1) for i in range(30)]
    feature_title += ["top1_style_id_{}".format(i+1) for i in range(30)]
    feature_title += ["interaction_flag_{}".format(i+1) for i in range(30)]
    for dist_name in ["actions", "pages", "styles"]:
        for duration in [30, 14, 7, 3, 2]:
            for candidate in range(category_to_num[dist_name]):
                feature_title.append("last_{}_cate_{}_dist_{}".format(dist_name, duration, candidate))
    feature_title += ["num_interactions_{}".format(i+1) for i in range(30)]
    for duration in [30, 14, 7, 3, 2]:
        feature_title += ["last_{}_{}_cnt".format(duration, cate) for cate in ["interaction", "song", "distinct"]]
    return feature_title, label_title


if __name__ == '__main__':
    parser = argparse.ArgumentParser("convert raw features to dataset for training...")
    parser.add_argument("--predict-dataset-file-name", type=str,
                        default="dataset/2021_1_data/total_features_to_predict.csv")
    parser.add_argument("--predict-features-file-name", type=str, default="dataset/2021_1_data/predict_features.csv")
    args = parser.parse_args()

    raw_data_file_name = args.predict_dataset_file_name
    assert os.path.exists(raw_data_file_name), "[ERROR] must provide an existed feature and label dataset"
    feature_file_name = args.predict_features_file_name
    feature_titles, label_titles = generate_titles()
    with open(raw_data_file_name, "r") as fp, open(feature_file_name, "w") as gp:
        gp.write("|".join(feature_titles) + "\n")
        print("[INFO] title = " + fp.readline().strip())
        for p in tqdm(fp, "[INFO] reading feature and labels from {} line-by-line ...".format(raw_data_file_name)):
            items = p.strip().split("|")
            device_index = int(items[0])
            category_features = CategoryFeature(items[1].split(','))
            distribution_features = DistributionFeatures(items[2].split(','))
            scale_features = [float(x) / 30.0 for x in items[3].split(',')]
            features = category_features.to_list()
            features.extend(distribution_features.to_list())
            features.extend(scale_features)
            gp.write("|".join(map(str, [device_index] + features)) + "\n")
