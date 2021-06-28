import json
import os
import torch
import transformers as tfs
import random
from torch import nn
from torch import optim
from tqdm import tqdm
from logger import Progbar
import Config
import Preprocess


if Config.BASE_MODEL_PATH == "":
    curdir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curdir, "model")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
else:
    model_path = Config.BASE_MODEL_PATH

# Bert预训练模型
FINETUNED_BERT_ENCODER_PATH = os.path.join(model_path, "finetuned_bert.bin")
POSITIVE_TRAIN_FILE_PATH = Preprocess.POSITIVE_TRAIN_FILE_PATH
POSITIVE_TRAIN_INFO_PATH = os.path.join(Preprocess.dataset_path, "positive_info.json")
UNLABELED_TRAIN_FILE_PATH = Preprocess.UNLABELED_TRAIN_FILE_PATH
PRETRAINED_BERT_ENCODER_PATH = Config.PRETRAINED_BERT_ENCODER_PATH
BERT_MODEL_SAVE_PATH = model_path
BATCH_SIZE = 2
EPOCH = 1


# 获取一个epoch需要的batch数
def get_steps_per_epoch(line_count, batch_size):
    return line_count // batch_size if line_count % batch_size == 0 else line_count // batch_size + 1


# 定义输入到Bert中的文本的格式,即标题,正文的组织形式
def prepare_sequence(title: str, body: str):
    return (title, body[:256] + "|" + body[-256:])


# 迭代器: 逐条读取数据并输出文本和标签
def get_text_and_label_index_iterator(input_path):
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in input_file:
            json_data = json.loads(line)
            text = prepare_sequence(json_data["title"], json_data["body"])
            label = json_data['label']

            yield text, label


# 迭代器: 生成一个batch的数据
def get_bert_iterator_batch(data_path, batch_size=32):
    keras_bert_iter = get_text_and_label_index_iterator(data_path)
    continue_iterator = True
    while continue_iterator:
        data_list = []
        for _ in range(batch_size):
            try:
                data = next(keras_bert_iter)
                data_list.append(data)
            except StopIteration:
                continue_iterator = False
        random.shuffle(data_list)

        text_list = []
        label_list = []

        for data in data_list:
            text, label = data
            text_list.append(text)
            label_list.append(label)

        yield text_list, label_list

    return False


class BertClassificationModel(nn.Module):
    """Bert分类器模型"""
    def __init__(self, model_path, predicted_size, hidden_size=768):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class = tfs.BertModel, tfs.BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        self.bert = model_class.from_pretrained(model_path)
        self.linear = nn.Linear(hidden_size, predicted_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=512,
                                                           pad_to_max_length=True)

        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        linear_output = self.dropout(self.linear(bert_cls_hidden_state).cuda()).cuda()
        return linear_output


def train_bert():
    if os.path.exists(POSITIVE_TRAIN_INFO_PATH):
        labels_set, total_num = Preprocess.get_label_set_and_sample_num(POSITIVE_TRAIN_INFO_PATH, True)
    else:
        print("Found no positive_info.json, please rerun the Preprocess.py.")
        exit()

    torch.cuda.set_device(0)

    print("Start training model...")
    # train the model
    steps = get_steps_per_epoch(total_num, BATCH_SIZE)

    bert_classifier_model = BertClassificationModel(PRETRAINED_BERT_ENCODER_PATH, len(labels_set))
    bert_classifier_model = bert_classifier_model.cuda()

    # 不同子网络设定不同的学习率
    Bert_model_param = []
    Bert_downstream_param = []
    number = 0
    for items, _ in bert_classifier_model.named_parameters():
        if "bert" in items:
            Bert_model_param.append(_)
        else:
            Bert_downstream_param.append(_)
        number += _.numel()
    param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                    {"params": Bert_downstream_param, "lr": 1e-4}]
    optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.6)
    criterion = nn.CrossEntropyLoss()
    bert_classifier_model.train()
    progbar = Progbar(target=steps)

    for epoch in range(EPOCH):
        model_save_path = os.path.join(BERT_MODEL_SAVE_PATH, "model_epoch{}.pkl".format(epoch))

        dataset_iterator = get_bert_iterator_batch(POSITIVE_TRAIN_FILE_PATH, BATCH_SIZE)

        for i, iteration in enumerate(dataset_iterator):
            # 清空梯度
            bert_classifier_model.zero_grad()
            text = iteration[0]
            labels = torch.tensor(iteration[1]).cuda()
            optimizer.zero_grad()
            output = bert_classifier_model(text)
            loss = criterion(output, labels).cuda()
            loss.backward()

            # 更新模型参数
            optimizer.step()
            # 学习率优化器计数
            StepLR.step()
            progbar.update(i + 1, None, None, [("train loss", loss.item()), ("bert_lr", optimizer.state_dict()["param_groups"][0]["lr"]), ("fc_lr", optimizer.state_dict()["param_groups"][1]["lr"])])

            if i == steps - 1:
                break

        # 保存完整的 BERT 分类器模型
        torch.save(bert_classifier_model, model_save_path)
        # 单独保存经 fune tune 的 BertEncoder模型
        torch.save(bert_classifier_model.bert, FINETUNED_BERT_ENCODER_PATH)
        print("epoch {} is over!\n".format(epoch))

    print("\nTraining is over!\n")
