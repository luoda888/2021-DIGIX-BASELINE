#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    impute the features and format features for lightGBM
    Input:
        1. train_features.csv: the valid features to train
        2. train_labels.csv: the related labels to train
    Output:
        1. lgb_train_dataset_X.bin.npy a formatted features
        2. lgb_train_dataset_y.bin.npy a formatted labels
        3. feature_selector.joblib a dumped file indicating the feature selector
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from joblib import dump, load
import os
import time


def load_features_and_labels(csv_features_file_name, csv_labels_file_name):
    """
    load features and labels from given file names
    :param csv_features_file_name: the file name of features
    :param csv_labels_file_name: the file name of labels
    :return: DataFrames of features and labels
    """
    t0 = time.time()
    features = pd.read_csv(csv_features_file_name, sep="|", header=0).values
    t1 = time.time()
    print("It takes {} seconds to load features from {}".format(t1 - t0, csv_features_file_name))

    t0 = time.time()
    labels = pd.read_csv(csv_labels_file_name, sep="|", header=0).values
    t1 = time.time()
    print("It takes {} seconds to load labels from {}".format(t1 - t0, csv_labels_file_name))
    return features, labels


def learn_or_load_imputer(dataset, impute_file_name="feature_imputer.joblib"):
    """
    learn a new feature selector if the given file does not exist
    :param dataset: the features dataset
    :param impute_file_name: the given filename of the serialized feature selector
    :return: the feature selector
    """
    if not os.path.exists(impute_file_name):
        t0 = time.time()
        selector = SimpleImputer(missing_values=0, strategy="most_frequent")
        selector.fit(dataset)
        t1 = time.time()
        print("It takes {} seconds to impute zeros for train set".format(t1 - t0))
        dump(selector, impute_file_name)
    else:
        selector = load(impute_file_name)
    return selector


def build_lgb_dataset(csv_features_file_name, csv_labels_file_name, lgb_dataset_base_name, selector_name):
    """
    impute the features, and format features and labels as a numpy format
    :param csv_features_file_name: the name of features
    :param csv_labels_file_name: the name of labels
    :param lgb_dataset_base_name: the name of target feature_and_label dataset
    :param selector_name: the name of feature selector
    """
    X, y = load_features_and_labels(csv_features_file_name, csv_labels_file_name)
    imputer = learn_or_load_imputer(X, selector_name)
    imputed_X = imputer.transform(X)
    print(imputed_X.shape)
    print("[INFO] split dataset into multi-labels")
    np.save(lgb_dataset_base_name.format("X"), imputed_X)
    np.save(lgb_dataset_base_name.format("y"), y)
    print("[INFO] generate done...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Impute features and generate formatted feature_and_labels...")
    parser.add_argument("--features-file-name", type=str, default="dataset/2021_1_data/train_features.csv")
    parser.add_argument("--labels-file-name", type=str, default="dataset/2021_1_data/train_labels.csv")
    parser.add_argument("--dataset-file-name-prefix", type=str, default="dataset/2021_1_data/lgb_train_dataset_")
    parser.add_argument("--preprocessor-file-name", type=str, default="dataset/2021_1_data/feature_selector.joblib")
    args = parser.parse_args()

    train_features_path = args.features_file_name
    train_labels_path = args.labels_file_name

    dataset_base_name = args.dataset_file_name_prefix + "{}"
    selector_file_name = args.preprocessor_file_name
    build_lgb_dataset(train_features_path, train_labels_path, dataset_base_name, selector_file_name)
