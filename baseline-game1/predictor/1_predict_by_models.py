#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    predict the final labels with given features
    Input:
        1. predict_features.csv: the valid features to predict
        2. feature_selector.joblib: the preprocessor to load
        3. model_cache_dir: the directory to save models
        4. model-name-prefix： the prefix about model names
        5. user_index_file_name: the file name for the relationships about user_id to user_index
    Output: result_file_name: the predicted results to submit
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from joblib import load

num_labels = 6


def load_device_map(device_id_to_index_file_name):
    """
    load the device_index to device_id from numbered dataset
    :param device_id_to_index_file_name: the specific file name about device_id to device_index
    :return: a dictionary about device_index to device_id
    """
    device_index_to_id = {}
    if not os.path.exists(device_id_to_index_file_name):
        return device_index_to_id
    with open(device_id_to_index_file_name, "r") as fp:
        print("[INFO] title = ", fp.readline().strip())
        for line in tqdm(fp, "[INFO] loading user_index to id csv..."):
            items = line.strip().split("|")
            device_index_to_id[items[1]] = items[0]
    return device_index_to_id


def load_trained_models(model_name_prefix):
    """
    load pre-trained models as given prefix
    :param model_name_prefix: the given prefix
    :return: a list of models for multi-tasks
    """
    models = []
    for label_id in range(num_labels):
        print("[INFO] load pre-trained model for label{}".format(label_id))
        model_name = model_name_prefix.format(label_id)
        if not os.path.exists(model_name):
            print("[ERROR] fail to find the predictor for {}".format(label_id))
            return []
        model = load(model_name)
        models.append(model)
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser("predict the labels based on features and trained models...")
    parser.add_argument("--features-file-name", type=str, default="dataset/2021_1_data/predict_features.csv")
    parser.add_argument("--preprocessor-file-name", type=str, default="dataset/2021_1_data/feature_selector.joblib")
    parser.add_argument("--model-cache-path", type=str, default="dataset/2021_1_data/lgb_model_label_")
    parser.add_argument("--user-index-file-name", type=str, default="dataset/2021_1_data/device_to_index.csv")
    parser.add_argument("--result-file-name", type=str, default="dataset/2021_1_data/submission.csv")
    args = parser.parse_args()

    assert os.path.exists(args.features_file_name), "[ERROR] must provide an non-empty feature file"
    assert os.path.exists(args.preprocessor_file_name), "[ERROR] must provide the preprocessor path"
    assert os.path.exists(args.user_index_file_name), "[ERROR] must provide an existed file about user_id to user_index"

    preprocessor = load(args.preprocessor_file_name)
    model_name_prefix = args.model_cache_path + "{}"
    models = load_trained_models(model_name_prefix)
    assert len(models) == num_labels, "[ERROR] must provide enough models for multi-tasks"

    device_index_to_id = load_device_map(args.user_index_file_name)
    with open(args.result_file_name, "w") as gp, open(args.features_file_name, "r") as fp:
        print("title = {}".format(fp.readline().strip()))
        gp.write("device_id,label_1d,label_2d,label_3d,label_7d,label_14d,label_30d\n")
        for line in tqdm(fp, "loading features and predict probability..."):
            items = line.strip().split("|")
            device_id = device_index_to_id.get(items[0], "0")
            features = np.asarray([[float(x) for x in items[1:]]], dtype=np.float)
            imputed = preprocessor.transform(features)
            props = [model.predict_proba(imputed)[0, 1] for model in models]
            trimmed_props = ["%.4f" % prop for prop in props]
            gp.write(",".join([device_id] + trimmed_props) + "\n")
