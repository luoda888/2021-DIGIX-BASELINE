import os
import random
import numpy as np
import json
import transformers as tfs
import torch
import Config
from torch import nn
from logger import Progbar
from tqdm import tqdm
import Preprocess
import Train_Bert

BATCH_SIZE = 2
POSITIVE_TRAIN_FILE_PATH = Preprocess.POSITIVE_TRAIN_FILE_PATH
POSITIVE_TRAIN_INFO_PATH = Preprocess.POSITIVE_TRAIN_INFO_PATH
UNLABELED_TRAIN_FILE_PATH = Preprocess.UNLABELED_TRAIN_FILE_PATH
BERT_TOKENZIER_PATH = Config.PRETRAINED_BERT_ENCODER_PATH
FINETUNED_BERT_ENCODER_PATH = Train_Bert.FINETUNED_BERT_ENCODER_PATH
PU_DATA_TEXT_SAVE_PATH = os.path.join(Preprocess.dataset_path, "PU_text.npy")
PU_DATA_LABEL_SAVE_PATH = os.path.join(Preprocess.dataset_path, "PU_label.npy")
STOP = False

# 获取一个epoch需要的batch数
def get_steps_per_epoch(line_count, batch_size):
    return line_count // batch_size if line_count % batch_size == 0 else line_count // batch_size + 1


# 获取数据集的标签集及其大小
def get_label_set_and_sample_num(config_path, sample_num=False):
    with open(config_path, "r", encoding="UTF-8") as input_file:
        json_data = json.loads(input_file.readline())
        if sample_num:
            return json_data["label_list"], json_data["total_num"]
        else:
            return json_data["label_list"]


# 定义输入到Bert中的文本的格式,即标题,正文的组织形式
def prepare_sequence(title: str, body: str):
    return (title, body[:256] + "|" + body[-256:])


# 迭代器: 逐条读取数据并输出文本和标签
def get_text_and_label_index_iterator(input_path):
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in input_file:
            json_data = json.loads(line)
            text = prepare_sequence(json_data["title"], json_data["body"])
            yield text


# 迭代器: 生成一个batch的数据
def get_bert_iterator_batch(data_path, batch_size=32):
    keras_bert_iter = get_text_and_label_index_iterator(data_path)
    continue_iterator = True
    while True:
        data_list = []
        for _ in range(batch_size):
            try:
                data = next(keras_bert_iter)
                data_list.append(data)
            except StopIteration:
                continue_iterator = False
                break
        random.shuffle(data_list)
        text_list = []
        if continue_iterator:
            for data in data_list:
                text_list.append(data)

            yield text_list
        else:
            return StopIteration


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


class MyBertEncoder(nn.Module):
    """自定义的Bert编码器"""
    def __init__(self, tokenizer_path, finetuned_bert_path):
        super(MyBertEncoder, self).__init__()
        model_class, tokenizer_class = tfs.BertModel, tfs.BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.bert = torch.load(finetuned_bert_path)

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=512, pad_to_max_length=True)

        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        return bert_cls_hidden_state


def build_pu_data():
    print("Start building PU data...")
    pos_data_iter = get_bert_iterator_batch(POSITIVE_TRAIN_FILE_PATH, batch_size=BATCH_SIZE)
    unlabeled_data_iter = get_bert_iterator_batch(UNLABELED_TRAIN_FILE_PATH, batch_size=BATCH_SIZE*2)

    torch.cuda.set_device(0)
    encoder = MyBertEncoder(BERT_TOKENZIER_PATH, FINETUNED_BERT_ENCODER_PATH)
    encoder.eval()
    X, y = None, None
    with torch.no_grad():
        i = 0
        for pos_batch, unlabeled_batch in tqdm(zip(pos_data_iter, unlabeled_data_iter)):
            encoded_pos = np.array(encoder(pos_batch).tolist())
            encoded_unlabeled = np.array(encoder(unlabeled_batch).tolist())
            if i == 0:
                X = np.concatenate([encoded_pos, encoded_unlabeled], axis=0)
                y = np.concatenate([np.full(shape=encoded_pos.shape[0], fill_value=1, dtype=np.int),
                                    np.full(shape=encoded_unlabeled.shape[0], fill_value=0, dtype=np.int)])
            else:
                X = np.concatenate([X, encoded_pos, encoded_unlabeled], axis=0)
                y = np.concatenate([y, np.full(shape=encoded_pos.shape[0], fill_value=1, dtype=np.int),
                                    np.full(shape=encoded_unlabeled.shape[0], fill_value=0, dtype=np.int)])

            i += 1

        np.save(PU_DATA_TEXT_SAVE_PATH, X)
        np.save(PU_DATA_LABEL_SAVE_PATH, y)
        print("PU data build successfully...")