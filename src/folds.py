#!/usr/bin/env python

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

labels = [
    'agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down',
    'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation',
    'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging',
    'slash_burn', 'water'
]


def encode_labels(df):
    tags = df['tags'].apply(lambda x: x.split(' ')).values
    mlb = MultiLabelBinarizer(labels)
    return mlb.fit_transform(tags)


def get_x_y():
    df_train = pd.read_csv('../input/train_v2.csv')
    y_train = encode_labels(df_train)
    return df_train['image_name'].values, y_train


def get_label_pack_complexity(df):
    num_labels_assigned_for_sample = df.sum(axis=1)
    mean_labels_by_label = {}

    for label in labels:
        ft = df[df[label] == 1].index
        mean_labels_by_label[label] = num_labels_assigned_for_sample.loc[
            ft].mean()  # noqa

    result = pd.DataFrame(
        list(mean_labels_by_label.items()),
        columns=['label', 'ratio'], )
    result.sort_values('ratio', ascending=False, inplace=True)
    return result


def make_balanced_folds(y, num_folds=5):

    df_y = pd.DataFrame(y, columns=labels)
    s_fold = pd.Series(index=df_y.index)

    label_counts = df_y.sum()
    mean_labels_by_label = get_label_pack_complexity(df_y)

    fold_counts = np.zeros((len(labels), num_folds))
    already_sampled = set()
    for label in mean_labels_by_label['label']:
        label_idx = labels.index(label)

        left_to_sample = df_y.index.difference(already_sampled)
        label_items = df_y[df_y[label] == 1].index
        to_sample = left_to_sample.intersection(label_items)

        print(label, to_sample.shape)
        print(fold_counts[label_idx, :])

        sample_index = to_sample.values

        np.random.seed(42 + label_idx)
        np.random.shuffle(sample_index)

        for item_idx in sample_index:
            fold_idx = np.argmin(fold_counts[label_idx, :])

            s_fold.loc[item_idx] = fold_idx

            fold_counts[:, fold_idx] += df_y.loc[item_idx, :]
            already_sampled.add(item_idx)

        print(fold_counts[label_idx, :])

        assert np.sum(fold_counts[label_idx, :]) == label_counts.loc[label]

    print(fold_counts)
    return s_fold.astype(int)


def make_splits():
    x, y = get_x_y()
    num_folds = 5
    s_fold = make_balanced_folds(y, num_folds)

    df = pd.DataFrame(y, columns=labels)
    df['image_name'] = x
    df['fold'] = s_fold

    for val_fold in reversed(range(num_folds)):
        ft = df['fold'] == val_fold
        train_df = df[~ft]
        val_df = df[ft]
        print(train_df.shape[0], val_df.shape[0])

        fold_dir = '../input/fold_{}'.format(num_folds - val_fold)
        os.makedirs(fold_dir, exist_ok=True)
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)


def main():
    make_splits()


if __name__ == '__main__':
    main()
