#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

NUM_DAYS = 30


class CategoryFeature(object):
    """
    the class of categorical features
    """

    def __init__(self, items):
        """
        the constructor to parse a list of items
        :param items: a list of items loaded from file
        """
        self.gender_id = int(items[0])
        self.age_id = int(items[1])
        self.city_id = int(items[2])
        self.phone_code = int(items[3])
        self.vip_flag = int(items[4])
        self.interaction_flags = [int(items[5 + 11 * n]) for n in range(NUM_DAYS)]
        self.channel_ids = [int(items[6 + 11 * n]) for n in range(NUM_DAYS)]
        self.top1_actions = [int(items[7 + 11 * n]) for n in range(NUM_DAYS)]
        self.top1_pages = [int(items[10 + 11 * n]) for n in range(NUM_DAYS)]
        self.top1_styles = [int(items[13 + 11 * n]) for n in range(NUM_DAYS)]

    def get_static_features(self):
        """
        rearrange the static features
        :return: the static features
        """
        return [[self.gender_id], [self.age_id], [self.city_id], [self.phone_code], [self.vip_flag]]

    def get_dynamic_features(self):
        """
        rearrange the behavior-based dynamic features
        :return: the dynamic features
        """
        return [self.channel_ids, self.top1_actions, self.top1_pages, self.top1_styles]

    def get_behavior_features(self):
        """
        rearrange the flags to indicate whether interacting daily
        :return: the interacted flags
        """
        return self.interaction_flags

    def to_list(self):
        """
        rearrange the features in a list
        :return: the total features in a list format
        """
        return [self.gender_id, self.age_id, self.city_id, self.phone_code, self.vip_flag] + \
               self.interaction_flags + self.channel_ids + self.top1_actions + self.top1_pages + self.top1_styles


def get_num_dist_based_name(name):
    """
    a function to make the reference from type-name to candidate_num
    :param name: the name of category
    :return: the number of candidates
    """
    name_to_num = {"actions": 9, "pages": 55, "styles": 6}
    return name_to_num[name]


def convert_to_dist(num_dist, dist_map):
    """
    normalized the distributions
    :param num_dist: the number of candidate values
    :param dist_map: the raw distributions
    :return: the normalized distributions in a list
    """
    epsilon = 1e-5
    dist_values = np.asarray([dist_map.get(idx, 0) for idx in range(num_dist)], dtype=np.float)
    dist_values /= (np.sum(dist_values) + epsilon)
    return dist_values.tolist()


def parse_distribution_from_str(dist_str):
    """
    convert the distribution in string-format to the dictionary
    :param dist_str: distribution in string-format
    :return: a dictionary as the distributions
    """
    if dist_str == '':
        return {}
    dist_lst = [x.split(':') for x in dist_str.split('#')]
    return dict((int(ps[0]), int(ps[1])) for ps in dist_lst)


class DistributionFeatures(object):
    """
    the class of the distribution features
    """

    def __init__(self, items):
        """
        constructor of DistributionFeatures
        :param items: the raw data
        """
        self.distribution_map = {}
        for i, cate in enumerate(["actions", "pages", "styles"]):
            self.distribution_map[cate] = [
                convert_to_dist(get_num_dist_based_name(cate), parse_distribution_from_str(items[i + 3 * n]))
                for n in range(5)
            ]

    def get_distribution(self, name):
        """
        return the distribution features
        :param name: the name of the category
        :return: the distribution fetures
        """
        return self.distribution_map.get(name, [])

    def to_list(self):
        """
        rearrange the features in a list
        :return: the total features in a list format
        """
        items = []
        for cate in ["actions", "pages", "styles"]:
            for n in range(5):
                items.extend(self.distribution_map[cate][n])
        return items
