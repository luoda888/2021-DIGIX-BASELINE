import json
import os
import torch
import numpy as np
import transformers as tfs
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
from torch import nn
from sklearn.externals import joblib
from Build_PU_data import MyBertEncoder
import Config
import Train_Bert
import Preprocess
import Train_PU_model

softmax = nn.Softmax(dim=1)
# Bert预训练模型
PRETRAINED_BERT_ENCODER_PATH = Config.PRETRAINED_BERT_ENCODER_PATH
FINETUNED_BERT_ENCODER_PATH = Train_Bert.FINETUNED_BERT_ENCODER_PATH
BERT_MODEL_SAVE_PATH = Train_Bert.BERT_MODEL_SAVE_PATH
PU_MODEL_SAVE_PATH = Train_PU_model.PU_MODEL_SAVE_PATH
TEST_FILE_PATH = Preprocess.PREPROCESSED_TEST_FILE_PATH

if Config.SUMMARY_OUTPUT_PATH == "":
    curdir = os.path.dirname(os.path.abspath(__file__))
    SUMMARY_OUTPUT_PATH = os.path.join(curdir, "submission.csv")
else:
    SUMMARY_OUTPUT_PATH = os.path.join(Config.SUMMARY_OUTPUT_PATH, "submission.csv")

INDEX = Preprocess.INDEX
MODEL_EPOCH = 5


# 获取数据集的标签集及其大小
def get_label_set_and_sample_num(config_path, sample_num=False):
    with open(config_path, "r", encoding="UTF-8") as input_file:
        json_data = json.loads(input_file.readline())
        if sample_num:
            return json_data["label_list"], json_data["total_num"]
        else:
            return json_data["label_list"]


# 生成数据集对应的标签集以及样本总数
def build_label_set_and_sample_num(input_paths, output_paths):
    label_set = set()
    sample_num = 0
    for input_path in input_paths:
        with open(input_path, 'r', encoding="utf-8") as input_file:
            for line in tqdm(input_file):
                json_data = json.loads(line)
                label_set.add(json_data["label"])
                sample_num += 1

    with open(output_paths, "w", encoding="UTF-8") as output_file:
        record = {"label_list": sorted(list(label_set)), "total_num": sample_num}
        json.dump(record, output_file, ensure_ascii=False)

        return record["label_list"], record["total_num"]


# 定义输入到Bert中的文本的格式,即标题,正文,source的组织形式
def prepare_sequence(title: str, body: str):
    return (title, body[:256] + "|" + body[-256:])


# 读取测试集数据, 这里使用 pd.read_json()
def read_test_file(input_path: str):
    test_df = pd.read_json(input_path, orient="records", lines=True)

    return test_df


def predict_with_pu(x, index, pu_classifier, bert_encoder, bert_classifier_model):
    text = prepare_sequence(x["title"], x["body"])

    encoded_pos = np.array(bert_encoder([text]).tolist())
    # 先使用 PU 预测是否为 "其他"
    pu_result = pu_classifier.predict(encoded_pos)
    if pu_result[0] < 0:
        predicted_label = "其他"
        proba = 0.5

    else:
        output = bert_classifier_model([text])
        predicted_proba = softmax(output).tolist()[0]
        predicted_index = np.argmax(predicted_proba)
        predicted_label = index[predicted_index]

        # 预测类别的预测概率
        proba = predicted_proba[predicted_index]

    return [predicted_label, round(proba, 2)]


# 结构化输出模型在测试集上的结果
def summary(test_df, output_path, pu_classifier, bert_encoder, bert_classifier_model):
    test_df[["predicted_label", "proba"]] = test_df.progress_apply(
        lambda x: pd.Series(predict_with_pu(x, INDEX, pu_classifier, bert_encoder, bert_classifier_model)), axis=1)

    # 提取id, predicted_label两列信息,并重命名列名, 最后输出到文件
    csv_data = test_df.loc[:, ["id", "predicted_label"]]
    csv_data.columns = ["id", "predict_doctype"]
    print("\n\n===================   The distribution of predictions   ===================\n")
    print(csv_data["predict_doctype"].value_counts())
    print("\n\n")
    csv_data.to_csv(output_path, index=0, line_terminator="\r\r\n")


class BertClassificationModel(nn.Module):
    """Bert模型支持两句输入..."""
    def __init__(self, model_path, predicted_size, hidden_size=768):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class = tfs.BertModel, tfs.BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        self.bert = model_class.from_pretrained(model_path)
        self.linear = nn.Linear(hidden_size, predicted_size)  # bert默认的隐藏单元数是768
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=512,
                                                           pad_to_max_length=True)

        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        linear_output = self.dropout(self.linear(bert_cls_hidden_state).cuda()).cuda()
        return linear_output


def joint_predictor():
    torch.cuda.set_device(0)
    fs = os.listdir(BERT_MODEL_SAVE_PATH)
    gs = list()
    for f in fs:
        if 'model_epoch' in f:
            gs.append(f)
    MODEL_EPOCH = max([int(x.split('.')[0].split('model_epoch')[-1]) for x in gs])
    model_save_path = os.path.join(BERT_MODEL_SAVE_PATH, "model_epoch{}.pkl".format(MODEL_EPOCH))
    print("Start evluation...")
    print("Load bert_classifier model path: ", model_save_path)
    print("Load PU_classifier model path: ", PU_MODEL_SAVE_PATH)
    test_df = read_test_file(TEST_FILE_PATH)

    # 读取 BERT 分类器模型
    bert_classifier_model = torch.load(model_save_path)
    bert_classifier_model = bert_classifier_model.cuda()
    bert_classifier_model.eval()

    with torch.no_grad():
        # 读取 PU 模型
        pu_classifier = joblib.load(PU_MODEL_SAVE_PATH)
        # 读取 fine tuned Bert Encoder模型
        bert_encoder = MyBertEncoder(PRETRAINED_BERT_ENCODER_PATH, FINETUNED_BERT_ENCODER_PATH)
        bert_encoder.eval()
        summary(test_df, SUMMARY_OUTPUT_PATH, pu_classifier, bert_encoder, bert_classifier_model)
    
    print("Evaluation done! Result has saved to: ", SUMMARY_OUTPUT_PATH)
