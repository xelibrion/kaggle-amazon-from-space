#!/usr/bin/env python

import os
import pandas as pd
import matplotlib.pyplot as plt

stages = set(['training', 'validation'])


def bin_batch(df):
    df.loc[:, 'batch_bin'] = pd.cut(df['batch'], bins=10, labels=False)
    return df


def bin_epoch(df):
    df.loc[:, 'epoch'] = df.loc[:, 'epoch'] + 0.1 * df.loc[:, 'batch_bin']
    return df


def plot_curves(df, filename='./curves.png'):
    fig = plt.figure(figsize=(10, 20))
    ax = plt.subplot(2, 1, 1)
    mean_df = df.groupby(['stage', 'epoch']).mean()
    p_df = pd.pivot_table(mean_df, columns='stage', index=['epoch'])
    p_df['loss'].plot(title='Mean loss by epoch', ax=ax)
    ax = plt.subplot(2, 1, 2)
    p_df['f2_score'].plot(title='Mean F2 by epoch', ax=ax)
    fig.savefig(filename)
    print(os.path.abspath(filename))


df = pd.read_json('./logs/events.json', lines=True)
df = df[df['stage'].isin(stages)]

plot_curves(df)

df = df.groupby('stage').apply(bin_batch).groupby('stage').apply(bin_epoch)
plot_curves(df, './curves_gran.png')
