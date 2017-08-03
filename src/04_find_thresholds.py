#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from tqdm import tqdm


def optimise_f2_thresholds(true_label,
                           prediction,
                           resolution=100,
                           init_thresholds=[0.2] * 17):

    thresholds = init_thresholds

    for i in tqdm(range(17), desc='Looking for optimal F2 thresholds'):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            thresholds[i] = i2
            score = fbeta_score(
                true_label,
                prediction > thresholds,
                beta=2,
                average='samples', )
            if score > best_score:
                best_i2 = i2
                best_score = score
        thresholds[i] = best_i2

    return thresholds, best_score


num_folds = 5

df_folds = [
    pd.read_csv('../input/fold_{}/val.csv'.format(x))
    for x in range(1, num_folds + 1)
]

df = pd.concat(df_folds)
true_label = df.iloc[:, 1:].values

fold_predictions = [
    np.load('./oof_predict_fold_{}.npy'.format(x))
    for x in range(1, num_folds + 1)
]

pred = np.vstack(fold_predictions)

thresholds, best_score = optimise_f2_thresholds(
    true_label, pred, resolution=1000)
print(thresholds)
print(best_score)
