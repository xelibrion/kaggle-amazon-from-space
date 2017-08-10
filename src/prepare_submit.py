#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from folds import labels

THRESHOLDS = np.array([
    0.159, 0.103, 0.166, 0.199, 0.259, 0.12, 0.067, 0.334, 0.215, 0.209, 0.198,
    0.125, 0.213, 0.203, 0.214, 0.283, 0.171
])


def mean_score(num_folds):
    np_folds = [
        np.load('./test_predict_fold_{}.npy'.format(x))
        for x in range(1, num_folds + 1)
    ]
    np_predict = np.mean(np_folds, axis=0)
    return (np_predict > THRESHOLDS).astype(int)


def voting(num_folds):
    np_folds = [
        np.load('./test_predict_fold_{}.npy'.format(x))
        for x in range(1, num_folds + 1)
    ]
    predicts = [x > THRESHOLDS for x in np_folds]
    votes = np.sum(predicts, axis=0)
    votes_threshold = int(num_folds / 2)
    return (votes > votes_threshold).astype(int)


def fit_binariser():
    mlb = MultiLabelBinarizer(labels)
    dtype = np.int if all(isinstance(c, int) for c in mlb.classes) else object
    mlb.classes_ = np.empty(len(mlb.classes), dtype=dtype)
    mlb.classes_[:] = mlb.classes
    return mlb


df_test = pd.read_csv('../input/sample_submission_v2.csv')

num_folds = 5

bin_label = mean_score(num_folds)

mlb = fit_binariser()
txt_label = mlb.inverse_transform(bin_label)
tags = [' '.join(x) for x in txt_label]

df_test['tags'] = tags

df_test.to_csv('submission.csv', index=False)
