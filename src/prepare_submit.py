#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from folds import labels

THRESHOLDS = np.array([
    0.185, 0.148, 0.164, 0.314, 0.193, 0.156, 0.05, 0.101, 0.192, 0.186, 0.169,
    0.213, 0.211, 0.183, 0.141, 0.289, 0.175
])


def mean_score(num_folds):
    np_folds = [
        np.load('./test_predict_fold_{}.npy'.format(x))
        for x in range(1, num_folds + 1)
    ]
    np_predict = np.mean(np_folds, axis=0)
    return (np_predict > THRESHOLDS).astype(int)


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
