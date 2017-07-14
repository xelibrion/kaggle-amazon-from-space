#!/usr/bin/env python

import functools
import os

import cv2
import numpy as np
import pandas as pd
from keras import backend as K, optimizers
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, Flatten
from keras.layers.core import Dropout
from keras.models import Model
from keras.utils.data_utils import Sequence
from sklearn.model_selection import train_test_split

from augment import expand_versions
from imgaug import augmenters as iaa
from densenet121 import DenseNet


def get_x_y():
    sample_size = int(os.environ.get('SAMPLE_SIZE', 0))

    img_train_dir = '../input/train-jpg'

    df_train = pd.read_csv('../input/train_v2.csv')
    df_train = df_train if not sample_size else df_train.sample(sample_size)
    df_train['image_name'] = df_train['image_name'].apply(
        lambda x: os.path.join(img_train_dir, "%s.jpg" % x))
    df_train.sort_values('image_name', inplace=True)
    y_train = encode_labels(df_train.copy())

    return df_train['image_name'].values, y_train.values


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


class AmazonTrainSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.X, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        slc = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.X[slc]
        batch_y = self.y[slc]

        raw_images = [cv2.imread(file_name, -1) for file_name in batch_x]
        aug_images = [expand_versions(x) for x in raw_images]
        batch_blob_x = np.array(
            functools.reduce(list.__add__, aug_images),
            dtype=np.float32, )

        repeat_count = batch_blob_x.shape[0] // len(raw_images)
        batch_y = np.repeat(batch_y, repeat_count, axis=0)

        return preprocess_input(batch_blob_x), np.array(batch_y)


class AmazonTestSequence(Sequence):
    def __init__(self, x_set, y_df, batch_size):
        self.X, self.y = x_set, y_df
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        slc = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.X[slc]
        batch_y = self.y[slc]

        batch_raw_x = [cv2.imread(file_name, -1) for file_name in batch_x]

        scissors = iaa.Crop(px=(16, 16, 16, 16), keep_size=False)
        batch_blob_x = [scissors.augment_image(x) for x in batch_raw_x]

        return preprocess_input(
            np.array(batch_blob_x, dtype=np.float32)), np.array(batch_y)


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


def get_model_vgg(num_classes):
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

    for l in base_model.layers:
        l.trainable = False

    x = base_model.layers[-1].output
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    predictions = Dense(
        num_classes,
        activation='sigmoid',
        name='predictions', )(x)

    return Model(inputs=base_model.input, outputs=predictions)


def get_model_resnet50(num_classes):
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))

    for l in base_model.layers:
        l.trainable = False

    x = base_model.layers[-1].output
    x = Flatten()(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(0.2, name='drop_fc1')(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dropout(0.2, name='drop_fc2')(x)

    predictions = Dense(
        num_classes,
        activation='sigmoid',
        name='predictions', )(x)

    return Model(inputs=base_model.input, outputs=predictions)


def get_model_densenet(num_classes):
    imagenet_weights = './densenet121_weights_tf.h5'

    base_model = DenseNet(
        reduction=0.5,
        classes=1000,
        weights_path=imagenet_weights, )
    base_model.layers.pop()
    base_model.layers.pop()

    trainable_threshold = [
        idx for idx, l in enumerate(base_model.layers)
        if isinstance(l, Conv2D)
    ]
    for l in base_model.layers[:trainable_threshold[-1] + 1]:
        l.trainable = False

    x = base_model.layers[-1].output
    x = Dense(num_classes, activation='sigmoid', name='fc_final')(x)

    return Model(inputs=base_model.input, outputs=x)


def lr_scheduler(epoch_idx):
    return 0.03
    # if epoch_idx == 0:
    #     return 0.03

    # if epoch_idx < 10:
    #     return 0.01

    # if epoch_idx > 10:
    #     return 0.001


def main():

    model_batch_size = int(os.environ.get('MODEL_BATCH_SIZE', 64))

    X, y = get_x_y()
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2)
    num_classes = y.shape[1]

    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_val), len(Y_val))

    train_seq = AmazonTrainSequence(
        X_train,
        Y_train,
        6, )

    val_seq = AmazonTestSequence(
        X_val,
        Y_val,
        model_batch_size, )

    # model = get_model_densenet(num_classes)
    model = get_model_resnet50(num_classes)
    # opt = optimizers.Adam()
    opt = optimizers.Nadam()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[fbeta])

    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=2, verbose=1),
        ModelCheckpoint(
            './xelibrion_weights_tf-{epoch}.h5',
            monitor='val_loss',
            save_best_only=True),
        # LearningRateScheduler(lr_scheduler),
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
        max_queue_size=100,
        use_multiprocessing=True,
        workers=2, )
    print("Training complete\n")


if __name__ == '__main__':
    main()
