import numpy as np
import os
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import Build_PU_data
import Train_Bert

PU_DATA_TEXT_SAVE_PATH = Build_PU_data.PU_DATA_TEXT_SAVE_PATH
PU_DATA_LABEL_SAVE_PATH = Build_PU_data.PU_DATA_LABEL_SAVE_PATH
PU_MODEL_SAVE_PATH = os.path.join(Train_Bert.model_path, "pu_model.bin")


class ElkanotoPuClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, hold_out_ratio=0.1):
        self.estimator = estimator
        # c is the constant proba that a example is positive, init to 1
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.estimator_fitted = False

    def __str__(self):
        return 'Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}'.format(
            self.estimator,
            self.c,
            self.estimator_fitted,
        )

    def split_hold_out(self, data):
        np.random.permutation(data)
        hold_out_size = int(np.ceil(data.shape[0] * self.hold_out_ratio))
        hold_out_part = data[:hold_out_size]
        rest_part = data[hold_out_size:]

        return hold_out_part, rest_part

    def fit(self, pos, unlabeled):
        # 打乱 pos 数据集, 按比例划分 hold_out 部分和非 hold_out 部分
        pos_hold_out, pos_rest = self.split_hold_out(pos)
        unlabeled_hold_out, unlabeled_rest = self.split_hold_out(unlabeled)

        all_rest = np.concatenate([pos_rest, unlabeled_rest], axis=0)
        all_rest_label = np.concatenate([np.full(shape=pos_rest.shape[0], fill_value=1, dtype=np.int),
                                             np.full(shape=unlabeled_rest.shape[0], fill_value=-1, dtype=np.int)])

        self.estimator.fit(all_rest, all_rest_label)

        # c is calculated based on holdout set predictions
        hold_out_predictions = self.estimator.predict_proba(pos_hold_out)
        hold_out_predictions = hold_out_predictions[:, 1]
        c = np.mean(hold_out_predictions)
        self.c = c
        self.estimator_fitted = True
        return self

    def predict_proba(self, X):
        if not self.estimator_fitted:
            raise NotFittedError(
                'The estimator must be fitted before calling predict_proba().'
            )
        probabilistic_predictions = self.estimator.predict_proba(X)
        probabilistic_predictions = probabilistic_predictions[:, 1]
        return probabilistic_predictions / self.c

    def predict(self, X, threshold=0.5):
        if not self.estimator_fitted:
            raise NotFittedError(
                'The estimator must be fitted before calling predict(...).'
            )
        return np.array([
            1.0 if p > threshold else -1.0
            for p in self.predict_proba(X)
        ])


def train_pu_model():
    print("\nStart fitting...")
    estimator = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        bootstrap=True,
        n_jobs=1,
    )
    pu_classifier = ElkanotoPuClassifier(estimator, hold_out_ratio=0.1)

    X = np.load(PU_DATA_TEXT_SAVE_PATH)
    y = np.load(PU_DATA_LABEL_SAVE_PATH)

    n_postive = (y == 1).sum()
    n_unlabeled = (y == 0).sum()
    print("total n_positive: ", n_postive)
    print("total n_unlabel:  ", n_unlabeled)
    # 随机筛选正样本和负样本
    # positive_random_index = np.random.choice(n_postive, RANDOM_POSITIVE_NUM)
    # unlabeled_random_index = np.random.choice(n_unlabeled, RANDOM_NEGATIVE_NUM)
    y_unlabel = np.ones(n_unlabeled)

    X_positive = X[y == 1]
    print("len of X_positive: ", X_positive.shape)
    y_positive_train = np.ones(n_postive)

    X_unlabel = X[y == 0]
    print("len of X_unlabeled: ", X_unlabel.shape)
    pu_classifier.fit(X_positive, X_unlabel)
    joblib.dump(pu_classifier, PU_MODEL_SAVE_PATH)
    print("Fitting done!")
