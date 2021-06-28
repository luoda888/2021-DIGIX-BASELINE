# A brief demo for menu identification

## 1. Brief introduction
This repo provides an example to complete menu identification task. Our example uses a two-step approach to perform this task, i.e., object detection and text recognition. For object detection, we adopt PSENet[1]. For text recognition, we adopt CRNN[2]. Please note that, in this example, we only use general text recognition approach and ignore the few-shot learning for special characters. New approaches can be explored by yourself.

## 2. Usage

### 2.1. preprocess

#### 2.1.1 folder structure
Firstly, we give the folder structure we used in this demo. The official_data folder contains the data downloaded from competition. The output folder contains some output during we perform this task. The checkpoints folder contains all trained models for object detection and text recognition. Furthermore, the tmp_data folder contains the temp data during predicting procedure.

```
menu_data
├──official_data
│     ├──train_image_common/*
│     ├──train_image_special/*
│     ├──train_label_common.json
│     ├──train_label_special.json
│     ├──test_image/*
├──output
│     ├──detector_test_output/*
│     ├──test_null.json
│     ├──test_submission.json
├──checkpoints
│     ├──detector/*
│     ├──recognizer/*
├──tmp_data
      ├──recognizer_txts/*
      ├──recognizer_images/*
```

#### 2.1.2 environment

required package:

```
tensorflow==1.14.0
opencv-python==4.1.0.25
bidict==0.19.0
yacs==0.1.8
Polygon==3.0.9
pyclipper==1.2.1
python3
```

running environment: CPU/GPU.

Furthermore, we use a C++ package to accelerate prediction speed for object detection task. To compile this package, please run the following command in the folder detector/postprocess
```
make
```

### 2.2. training object detection model

After putting downloaded files to official_data folder, please set the path in the detector/config/resnet50.yaml. Then run the following command:
```
python detector/train.py
```

After learning procedure, a trained model will be saved at: /path/to/menu-data/checkpoints/detector/.

### 2.3. training text recognition model

To train text recognition model, we first perform some pre-processing. Please run the command:
```
python recognizer/tools/extract_train_data.py \ 
    --save_train_image_path /path/to/tmp_data/recognizer_images \
    --save_train_txt_path  /path/to/tmp_data/recognizer_txts \
    --train_image_common_root_path /path/to/official_data/train_image_common \
    --common_label_json_file /path/to/official_data/train_label_common.json \
    --train_image_special_root_path /path/to/official_data/train_image_special \
    --special_label_json_file /path/to/official_data/train_label_special.json
```
Then, please run the command:
```
python recognizer/tools/from_text_to_label.py \
    --src_train_file_path /path/to/tmp_data/recognizer_txts/train.txt \
    --dst_train_file_path /path/to/tmp_data/recognizer_txts/real_train.txt \
    --dictionary_file_path recognizer/tools/dictionary/chars.txt
```

Here, we use chars.txt to store all characters. If you change the param dictionary_file_path, please change the path param in the recognizer/tools/config.py as well.

Then, run the following command:
```
python recognizer/train.py \
    --model_save_dir /path/to/checkpoints/recognizer \
    --log_dir /path/to/output/recognizer_log \
    --image_dir /path/to/tmp_data/recognizer_images \
    --txt_dir /path/to/tmp_data/recognizer_txts
```

Then we can get a trained model which stored at /path/to/menu-data/checkpoint/recognizer/

### 2.4. prediction procedure
To perform prediction procedure, please put test images into folder /path/to/menu-data/official_data.
#### 2.4.1. object detection

Set the params in the detector/config/resnet50.yaml and run the following command:
```
python detector/predict.py
```
The output of object detection model will be stored at /path/to/menu-data/output/detector_test_output. And detector will generate a null json file test_null.json. These files will be used as the input for text recognition.
#### 2.4.2. text recognition

Set the params in the recognizer/predict.py and run the following command:
```
python recognizer/predict.py \
    --char_path recognizer/tools/dictionary/chars.txt \
    --model_path /path/to/checkpoints/recognizer/*.h5 \
    --null_json_path /path/to/output/test_null.json \
    --test_image_path /path/to/output/detector_test_output \
    --submission_path /path/to/output/test_submission.json
```

The generated file test_submission.json can be submitted to the competition.

## 3. Reference

- [1]. Wang, Wenhai, et al. Shape robust text detection with progressive scale expansion network. CVPR. 2019.
- [2]. Shi, Baoguang, Xiang Bai, and Cong Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. TPAMI, 2016. 39.11: 2298-2304.