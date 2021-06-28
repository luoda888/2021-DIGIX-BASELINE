import numpy as np
import tensorflow as tf
import os

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64list_feature(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


# Example序列化成字节字符串
def serialize_example(watch, share, age, province, city, citylevel, devicename,videoid,videoscore,videoduration ):
    # 注意我们需要按照格式来进行数据的组装，这里的dict便按照指定Schema构造了一条Example
    feature = {
      'watch': _int64_feature(watch),
      'share': _int64_feature(share),
       'age':_int64_feature(age),
      'province': _int64_feature(province),
      'city': _int64_feature(city),
      'citylevel': _int64_feature(citylevel),
       'devicename': _int64_feature(devicename),
        'videoid': _int64_feature(videoid),
        'videoscore': _float_feature(videoscore),
        'videoduration':_float_feature(videoduration)
    }
    # 调用相关api将Example序列化为字节字符串
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



writer = tf.python_io.TFRecordWriter(r"data\samples\train-0000.tfrecord")

file_path = r"data\rawdata"
file_list = os.listdir(file_path)
for file_name in file_list:
    index_file = os.path.join(file_path,file_name)
    index_list = open(index_file, "r").readlines()[1:]    # 读取索引文件，去掉首行
    print("总记录数:{}".format(len(index_list)))
    i = 0

    for line in index_list:
        feat = []
        feats = []
        print("第{}条记录".format(i))
        for l in line.rstrip('\n').split('\t'):
            if l != '':
                feats.append(l)
        print(feats)
        if len(feats) == 10:
            for j in range(len(feats)):
                if j<=7 and feats[j] is not None:
                    feat.append(int(feats[j]))
                elif feats[j]!='':
                    feat.append(float(feats[j]))
            writer.write(serialize_example(*feat))
        i = i + 1

writer.close()