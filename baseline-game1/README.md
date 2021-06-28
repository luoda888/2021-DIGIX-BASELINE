[toc]



### Struct of codes

>+--- feature_builder
>|   +--- 0_generate_features.py
>+--- predictor
>|   +--- 0_retrain_and_predict.py
>|   +--- Utils.py
>+--- preprocess
>|   +--- 0_numbered_items.py
>|   +--- 1_format_active_logs.py
>|   +--- 2_format_user_info.py
>|   +--- 3_format_music_info.py
>|   +--- 4_format_user_behavior.py
>|   +--- utils.py
>+--- README.md
>+--- requirements.txt
>+--- trainer
>|   +--- 0_reformat_features.py
>|   +--- 1_impute_and_convert_to_lgb.py
>|   +--- 2_run_lightgbm.py
>|   +--- Features.py
>|   +--- Utils.py



### Requirements

as the `requirements.txt`



### Preprocessing

```shell
python -u preprocess/0_numbered_items.py --user-info-file-name "dataset/2021_1_data/2_user_info.csv" --song-info-file-name "dataset/2021_1_data/3_music_info.csv" --user-numbered-file-name "dataset/2021_1_data/device_to_index.csv" --song-numbered-file-name "dataset/2021_1_data/music_to_index.csv"

python -u preprocess/1_format_active_logs.py --active-log-file-name "dataset/2021_1_data/1_device_active.csv" --train-log-file-name "dataset/2021_1_data/active_logs_to_train.csv" --predict-log-file-name "dataset/2021_1_data/active_logs_to_predict.csv" --user-index-file-name "dataset/2021_1_data/device_to_index.csv"

python -u preprocess/2_format_user_info.py --device-info-file-name "dataset/2021_1_data/2_user_info.csv" --user-index-file-name "dataset/2021_1_data/device_to_index.csv" --user-side-info-file-name "dataset/2021_1_data/user_with_side_information.csv" --phone-index-file-name "dataset/2021_1_data/phone_id_to_index.csv" --city-index-file-name "dataset/2021_1_data/city_id_to_index.csv"

python -u preprocess/3_format_music_info.py --music-info-file-name "dataset/2021_1_data/3_music_info.csv" --artist-info-file-name "dataset/2021_1_data/5_artist_info.csv" --song-index-file-name "dataset/2021_1_data/music_to_index.csv" --song-info-file-name "dataset/2021_1_data/song_with_side_information.csv" --style-index-file-name "dataset/2021_1_data/style_name_to_index.csv"

python -u preprocess/4_format_user_behavior.py --behaviors-file-name "dataset/2021_1_data/4_user_behavior.csv" --user-index-file-name "dataset/2021_1_data/device_to_index.csv" --song-index-file-name "dataset/2021_1_data/music_to_index.csv" --behavior-feature-file-name "dataset/2021_1_data/user_behaviors_with_static_info.csv"
```



### Generate features

```shell
python -u feature_builder/0_generate_features.py --user-static-features "dataset/2021_1_data/user_with_side_information.csv" --song-static-features "dataset/2021_1_data/song_with_side_information.csv" --user-behaviors-features "dataset/2021_1_data/user_behaviors_with_static_info.csv" --training-active-logs "dataset/2021_1_data/active_logs_to_train.csv" --training-dataset "dataset/2021_1_data/total_features_to_train.csv" --predicting-active-logs "dataset/2021_1_data/active_logs_to_predict.csv" --predicting-dataset "dataset/2021_1_data/total_features_to_predict.csv" --temp-dir "dataset/2021_1_data" --bad-case "dataset/2021_1_data/missing_behaviors.csv"
```



### Train models

#### Extract valid features

```shell
python -u trainer/0_extract_valid_features.py --train-dataset-file-name "dataset/2021_1_data/total_features_to_train.csv" --train-features-file-name "dataset/2021_1_data/train_features.csv" --train-labels-file-name "dataset/2021_1_data/train_labels.csv"
```

#### Impute and restore as numpy format

```shell
python -u trainer/1_impute_features.py --features-file-name "dataset/2021_1_data/train_features.csv" --labels-file-name "dataset/2021_1_data/train_labels.csv" --dataset-file-name-prefix "dataset/2021_1_data/lgb_train_dataset_" --preprocessor-file-name "dataset/2021_1_data/feature_selector.joblib"
```

#### train models

```shell
python -u trainer/2_run_lightgbm.py --features-file-name "dataset/2021_1_data/lgb_train_dataset_X.npy" --labels-file-name "dataset/2021_1_data/lgb_train_dataset_y.npy" --model-cache-path "dataset/2021_1_data/lgb_model_label_"
```



### Predict the next labels

```shell
python -u predictor/0_extract_features_for_predict.py --predict-dataset-file-name "dataset/2021_1_data/total_features_to_predict.csv" --predict-features-file-name "dataset/2021_1_data/predict_features.csv"

python -u predictor/1_predict_by_models.py --features-file-name "dataset/2021_1_data/predict_features.csv" --preprocessor-file-name "dataset/2021_1_data/feature_selector.joblib" --model-cache-path "dataset/2021_1_data/lgb_model_label_" --user-index-file-name "dataset/2021_1_data/device_to_index.csv" --result-file-name "dataset/2021_1_data/submission.csv"
```

