#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    train a lightGBM Classifier for each label based on imputed features
    Input:
        1. lgb_train_dataset_X.bin.npy a formatted features
        2. lgb_train_dataset_y.bin.npy a formatted labels
    Output:
        lgb_model_label_{}, indicate the trained predictor about label1~6
"""

import os
import argparse
import numpy as np
import lightgbm as lgb
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

num_labels = 6
category_indices = [0, 1, 2, 3, 4] + [idx + 5 for idx in range(150)]


def eval_lgb_model(X, y, lgb_model=None):
    """
    ecalute the lightGBM trained model via given X and y
    :param X: the features dataset
    :param y: the labels dataset
    :param lgb_model: the given model to evaluate
    :return: the auc score
    """
    props = lgb_model.predict_proba(X)
    return roc_auc_score(y, props[:, 1])


def model_training(train_X, train_y, valid_X, valid_y, tuner_params=None):
    """
    train a LightGBM classifier by tuner parameters
    :param train_X: the training features
    :param train_y: the training labels
    :param valid_X: the validation features
    :param valid_y: the validation labels
    :param tuner_params: the given parameters to tuning model
    :return: a trained lgb model and auc score in valid-set
    """
    if tuner_params is None:
        lgb_model = lgb.LGBMClassifier(max_depth=8, num_leaves=64, objective="binary", learning_rate=0.05, n_jobs=8,
                                       feature_fraction=0.7, min_child_weight=0.001, reg_alpha=0.001, reg_lambda=6,
                                       n_estimators=1000)
    else:
        lgb_model = lgb.LGBMClassifier(max_depth=tuner_params[0], num_leaves=tuner_params[1], objective="binary",
                                       learning_rate=0.05, n_jobs=8, feature_fraction=tuner_params[2],
                                       min_child_weight=0.001, reg_alpha=0.001, reg_lambda=6, n_estimators=1000)
    lgb_model.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], eval_metric="auc",
                  early_stopping_rounds=20, verbose=20, categorical_feature=category_indices)
    valid_props = lgb_model.predict_proba(valid_X)[:, 1]
    valid_auc_score = roc_auc_score(valid_y, valid_props)
    return lgb_model, valid_auc_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train lightGBM classifier based on dataset...")
    parser.add_argument("--features-file-name", type=str, default="dataset/2021_1_data/lgb_train_dataset_X.npy")
    parser.add_argument("--labels-file-name", type=str, default="dataset/2021_1_data/lgb_train_dataset_y.npy")
    parser.add_argument("--model-cache-path", type=str, default="dataset/2021_1_data/lgb_model_label_")
    args = parser.parse_args()

    train_features_path = args.features_file_name
    train_labels_path = args.labels_file_name
    model_cache_path = args.model_cache_path + "{}"

    weighted_auc = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    X0 = np.load(train_features_path)
    y0 = np.load(train_labels_path)
    total_size = y0.shape[0]
    test_size = int(total_size * 0.2)
    train_X, test_X, train_y, test_y = train_test_split(X0, y0, train_size=total_size - test_size, test_size=test_size)
    print("[INFO] trainX size = {}".format(train_X.shape))
    print("[INFO] trainY size = {}".format(train_y.shape))
    print("[INFO] trainX size = {}".format(test_X.shape))
    print("[INFO] trainY size = {}".format(test_y.shape))
    params = [[8, 40, 0.6], [8, 40, 0.6], [8, 40, 0.6], [6, 40, 0.6], [8, 40, 0.6], [8, 30, 0.6]]
    for label_id in range(num_labels):
        model, valid_auc = model_training(
            train_X, train_y[:, label_id], test_X, test_y[:, label_id], params[label_id])
        print("best valid-auc in {} = {}".format(label_id, valid_auc))
        dump(model, model_cache_path.format(label_id))
        weighted_auc[label_id] *= valid_auc
    print("weighted auc = {}".format(np.sum(weighted_auc)))
