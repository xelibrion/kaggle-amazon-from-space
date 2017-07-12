#!/usr/bin/env python

import os
from densenet121 import DenseNet
import numpy as np
import pandas as pd
import cv2
import platform

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             LearningRateScheduler, TensorBoard, Callback)
from keras.utils.data_utils import Sequence

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Activation
from keras import backend as K

train_dir = '../input/train-jpg'
model_batch_size = os.environ.get('MODEL_BATCH_SIZE', 64)


def encode_labels(df):
    df.set_index('image_name', inplace=True)
    df_tag_cols = df['tags'].str.split(' ').apply(pd.Series).reset_index()
    df_tag_melted = pd.melt(
        df_tag_cols,
        id_vars='image_name',
        value_name='tag',
        var_name='present', )
    df_tag_melted['present'] = df_tag_melted['present'].astype(int)
    df_tag_melted.loc[~pd.isnull(df_tag_melted['tag']), 'present'] = 1

    df_y = pd.pivot_table(
        df_tag_melted,
        index='image_name',
        columns='tag',
        fill_value=0, )

    df_y.columns = df_y.columns.droplevel(0)
    df_y[df_y.columns] = df_y[df_y.columns].astype(np.uint8)
    return df_y


class AmazonSequence(Sequence):
    def __init__(self, x_set, y_df, batch_size, img_size=224):
        self.X, self.y = x_set, y_df
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y.loc[batch_x].values

        img_size = (self.img_size, self.img_size)
        batch_blob_x = [
            cv2.resize(cv2.imread(file_name, -1), img_size)
            for file_name in batch_x
        ]

        return np.array(batch_blob_x), np.array(batch_y)


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta**2
    return (beta_squared + 1) * (precision * recall) / (
        beta_squared * precision + recall + K.epsilon())


df_train = pd.read_csv('../input/train_v2.csv')
df_train['image_name'] = df_train['image_name'].apply(
    lambda x: os.path.join(train_dir, "%s.jpg" % x))
df_train.sort_values('image_name', inplace=True)
y_train = encode_labels(df_train.copy())
num_classes = y_train.shape[1]

X_train, X_val, Y_train, Y_val = train_test_split(
    df_train['image_name'].values,
    y_train,
    test_size=0.2, )

print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_val), len(Y_val))

train_seq = AmazonSequence(
    X_train,
    Y_train,
    model_batch_size, )

val_seq = AmazonSequence(
    X_val,
    Y_val,
    model_batch_size, )

imagenet_weights = './densenet121_weights_tf.h5'

base_model = DenseNet(
    reduction=0.5, classes=1000, weights_path=imagenet_weights)
base_model.layers.pop()
base_model.layers.pop()
for l in base_model.layers:
    l.trainable = False

x = base_model.layers[-1].output
x = Dense(num_classes, name='fc6')(x)
predictions = Activation('sigmoid', name='prob')(x)

model = Model(inputs=base_model.input, outputs=predictions)
opt = optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[fbeta])


def lr_scheduler(epoch_idx):
    if epoch_idx == 0:
        return 0.003

    if epoch_idx < 10:
        return 0.01

    if epoch_idx > 10:
        return 0.001


callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=1),
    ModelCheckpoint(
        './xelibrion_weights_tf-%d.h5',
        monitor='val_loss',
        save_best_only=True),
    LearningRateScheduler(lr_scheduler),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1, )
]

print("Starting training")
model.fit_generator(
    train_seq,
    epochs=60,
    steps_per_epoch=len(train_seq),
    validation_data=val_seq,
    validation_steps=len(val_seq),
    verbose=1,
    callbacks=callbacks,
    max_queue_size=3 * model_batch_size,
    use_multiprocessing=True,
    workers=2, )
print("Training complete\n")
