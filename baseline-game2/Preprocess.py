import json
import os
import pandas as pd
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
import Config

if Config.BASE_DATASET_PATH == "":
    curdir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(curdir, "dataset")
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
else:
    dataset_path = Config.BASE_DATASET_PATH

RAW_TRAIN_FILE_PATH = os.path.join(dataset_path, "doc_quality_data_train_1000.json")
RAW_TEST_FILE_PATH = os.path.join(dataset_path, "doc_quality_data_test_1000.json")
PREPROCESSED_TRAIN_FILE_PATH = os.path.join(dataset_path, "preprocessed_train.json")
PREPROCESSED_TEST_FILE_PATH = os.path.join(dataset_path, "preprocessed_test.json")
POSITIVE_TRAIN_FILE_PATH = os.path.join(dataset_path, "postive_train.json")
POSITIVE_TRAIN_INFO_PATH = os.path.join(dataset_path, "positive_info.json")
UNLABELED_TRAIN_FILE_PATH = os.path.join(dataset_path, "unlabeled_train.json")
# 优质类别索引列表
INDEX = ['人物专栏', '作品分析', '情感解读', '推荐文', '攻略文', '治愈系文章', '深度事件', '物品评测', '科普知识文', '行业解读']

# 获取数据集的标签集及其大小
def get_label_set_and_sample_num(config_path, sample_num=False):
    with open(config_path, "r", encoding="UTF-8") as input_file:
        json_data = json.loads(input_file.readline())
        if sample_num:
            return json_data["label_list"], json_data["total_num"]
        else:
            return json_data["label_list"]


# 生成数据集对应的标签集以及样本总数
def build_label_set_and_sample_num(input_path, output_path):
    label_set = set()
    sample_num = 0
    
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            json_data = json.loads(line)
            label_set.add(json_data["label"])
            sample_num += 1
            
    with open(output_path, "w", encoding="UTF-8") as output_file:
        record = {"label_list": sorted(list(label_set)), "total_num": sample_num}
        json.dump(record, output_file, ensure_ascii=False)

        return record["label_list"], record["total_num"]


def get_sentences_list(raw_text: str):
    return [s for s in BeautifulSoup(raw_text, 'html.parser')._all_strings()]


def check_length(length_list):
    sum_length = sum(length_list)
    if sum_length < 510:
        return sum_length
    return 510


# 去除空白字符, 从数据集遍历代码中移至此处
def remove_symbol(string: str):
    return string.replace('\t', '').replace('\n', '').replace('\r', '')


def check_duplicate_title(input_path, output_path):
    duplicate = 0
    no_html = 0
    no_duplicate = 0
    print("Processing File: ", input_path)
    with open(input_path, "r", encoding='utf-8') as file, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(file):
            json_data = json.loads(line)
            title = json_data["title"]
            body = get_sentences_list(json_data["body"])
            title_length = len(title)

            # 正文中不含HTML标签
            if len(body) == 1:
                no_html += 1
                tmp_body = body[0]
                # 注意,这边re.sub的pattern使用了re.escape()
                # 是为了转译title中存在的会被re视为元字符的字符(例如"?"","*")
                # 事实上相当于"\".join(title)[将所有字符转译为普通字符]
                new_body = re.sub("(原标题：)?" + re.escape(title), "", tmp_body)
                new_body_length = len(new_body)

                if new_body_length == len(tmp_body):
                    no_duplicate += 1
                else:
                    duplicate += 1

            # 正文中包含HTML标签
            else:
                i = 0
                # 检查 标题是否出现在前两个元素中 (有可能存在标签<p class=\"ori_titlesource\">,会有"原标题: title"的情况出现)
                for sentence in body[:2]:
                    if title in sentence:
                        i += 1

                new_body = "".join(body[i:])

                if i > 0:
                    duplicate += 1
                else:
                    no_duplicate += 1

            rm_whites_body = remove_symbol(new_body)
            rm_whites_title = remove_symbol(title)

            json_data["body"] = rm_whites_body
            json_data["title"] = rm_whites_title
            json_data["length"] = check_length([len(rm_whites_body), len(rm_whites_title)])
            json.dump(json_data, outfile, ensure_ascii=False)
            outfile.write("\n")

    print("duplicate: {}\t no_html: {}, no_duplicate: {}\n".format(duplicate, no_html, no_duplicate))


def index_data_pd(index, input_path, output_path1, output_path2):
    print(input_path)
    df_data = pd.read_json(input_path, orient="records", lines=True)
    # 处理已标注数据
    df_data_labeled = df_data[df_data["doctype"] != ""]
    df_data_labeled = df_data_labeled.sample(frac=1.0)

    df_data_labeled["label"] = df_data_labeled.apply(lambda x: index.index(x["doctype"]), axis=1, raw=False)
    
    print("\n\n===================   The distribution of Positive train data   ===================\n")
    print(df_data_labeled["label"].value_counts())
    print("\n\n")
    df_data_labeled = df_data_labeled.drop(columns=["category"])
    # 单独保存已标注数据
    df_data_labeled.to_json(output_path1, orient="records", lines=True, force_ascii=False)

    # 处理未标注数据
    df_data_unlabeled = df_data[df_data["doctype"] == ""]
    df_data_unlabeled = df_data_unlabeled.sample(frac=1.0)
    df_data_unlabeled = df_data_unlabeled.drop(columns=["category"])
    # 单独保存未标注数据
    df_data_unlabeled.to_json(output_path2, orient="records", lines=True, force_ascii=False)


def preprocess():
    # 清除训练集、测试集的文章正文中可能存在的标题
    check_duplicate_title(RAW_TRAIN_FILE_PATH, PREPROCESSED_TRAIN_FILE_PATH)
    check_duplicate_title(RAW_TEST_FILE_PATH, PREPROCESSED_TEST_FILE_PATH)
    # 对训练集中带标记的样本索引化标签
    index_data_pd(INDEX, PREPROCESSED_TRAIN_FILE_PATH,
                  POSITIVE_TRAIN_FILE_PATH, UNLABELED_TRAIN_FILE_PATH)

    if os.path.exists(POSITIVE_TRAIN_INFO_PATH):
        labels_set, total_num = get_label_set_and_sample_num(POSITIVE_TRAIN_INFO_PATH, True)
    else:
        labels_set, total_num = build_label_set_and_sample_num(POSITIVE_TRAIN_FILE_PATH, POSITIVE_TRAIN_INFO_PATH)
    print("Preprocess done!")


