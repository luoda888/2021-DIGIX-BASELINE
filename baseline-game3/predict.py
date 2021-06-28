
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib import predictor
from tensorflow import feature_column as fc

model_name = './model_output'
my_feature_columns = []

def load_model():
    # 读取模型
    model_time = str(max([int(i) for i in os.listdir(model_name) if len(i)==10]))
    print(os.path.join(model_name,model_time))
    model = predictor.from_saved_model(os.path.join(model_name,model_time))
    return model


if __name__ == '__main__':


    model = load_model()

    inputs ={
    'age': [0,1,2],
    'province': [0,0,0],
    'city':[0,0,0],
    'citylevel': [0,0,0],
    'devicename': [0,1,2],
    'videoid': [0,1,2],
    'videoscore': [6.2,3.3,8.5],
    'videoduration': [0,6847,200]
    }
    # 将输入数据转换成序列化后的 Example 字符串。
    examples = []
    for i in range(1):
        feature = {}
        for col in inputs:
            if col=="videoscore" or col=="videoduration":
                feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[inputs[col][i]]))
            else:
                feature[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[inputs[col][i]]))
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature
                )
            )
            examples.append(example.SerializeToString())


    predictions = model({'examples': examples})
    print(predictions)

