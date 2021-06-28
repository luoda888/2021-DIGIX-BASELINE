# A Baseline Model for Article Quality Evaluation

This repository provides the code of a baseline model developed for Article Quality Evaluation. This baseline model is conducted mainly based on Bert and PU Learning technology. Specifically, given a article, Bert is used to encoder it and output corresponding representation. And PU Learning technology is introduced when training a binary classifier which can predict whether a article is high quality or not.

Briefly speaking, this baseline model implements the following strategy to perform Article Quality Evaluation:
1. Build a positive sample dataset(P, 73K samples) by extracting all the labeled samples from raw training dataset and the remaining samples can be named as Unlabeled sample dataset(N, 500K samples). 
2. Use P to train a 10-class classifier base on Bert and LSTM.
3. Feed the trained 10-class classifier with U and filter the samples which probs are < 0.8, and this filtered samples can be regarded roughly as reliable negative samples(RN, 130K samples)
4. Randomly extract certain amount of samples from P and RN and use them (50K samples) to train a binary classifier;
5. At stage of inference, the sample for testing is fed to the binary classifier firstly to predict whether it is high-quality. If not, it will be labeled as "others", and if yes, it will be further fed to the 10-class classifier to predict which one of 10 classes it belongs to.

To run the model, you just need to configure the hyper-parameters listed in Config.py according to your own environment and then execute main.py.