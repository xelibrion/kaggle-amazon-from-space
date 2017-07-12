#!/usr/bin/env python

import os
from densenet121 import DenseNet
import numpy as np
import pandas as pd
import cv2
import platform

from tqdm import tqdm
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             LearningRateScheduler, TensorBoard, Callback)

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator

train_dir = ('../input/train-jpg'
             if platform.system() == 'Linux' else '../input/train-sample')
model_batch_size = os.environ.get('MODEL_BATCH_SIZE', 256)


def flatten(l):
    return [item for sublist in l for item in sublist]


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


x_train = []
x_test = []

df_train = pd.read_csv('../input/train_v2.csv')
df_test = pd.read_csv('../input/sample_submission_v2.csv')

y_train = encode_labels(df_train)
num_classes = y_train.shape[1]
img_size = 224

# print("Resizing test set")
# for f, tags in tqdm(df_test.values, miniters=1000):
#     img = cv2.imread('../input/test-tif-v2/{}.tif'.format(f), -1)
#     x_test.append(cv2.resize(img, (img_size, img_size)))
# x_test = np.array(x_test, np.float32) / 255.

print("Resizing train set")
for image_name, tags in tqdm(df_train.values, miniters=1000):
    img_path = '{}/{}.jpg'.format(train_dir, image_name)
    if not os.path.exists(img_path):
        continue
    # https://stackoverflow.com/questions/37512119/resize-transparent-image-in-opencv-python-cv2
    # If you load a 4 channel image, the flag -1 indicates that the image
    # is loaded unchanged, so you can load and split all 4 channels directly.
    img = cv2.imread(img_path, -1)
    x_train.append(cv2.resize(img, (img_size, img_size)))

x_train = np.array(x_train, np.float32) / 255.

print("x_train shape:")
print(x_train.shape)
print("y_train shape:")
print(y_train.shape)

X_train, X_val, Y_train, Y_val = train_test_split(
    x_train, y_train, test_size=0.2)

print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_val), len(Y_val))

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
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


def lr_scheduler(epoch_idx):
    if epoch_idx == 0:
        return 0.003

    if epoch_idx < 10:
        return 0.01

    if epoch_idx > 10:
        return 0.001


class Fbeta(Callback):
    def on_train_begin(self, logs={}):
        self.fbeta = []

    def on_epoch_end(self, epoch, logs={}):
        p_valid = self.model.predict(self.validation_data[0])
        y_val = self.validation_data[1]
        f_beta = fbeta_score(
            y_val,
            np.array(p_valid) > 0.5,
            beta=2,
            average='samples', )
        self.fbeta.append(f_beta)
        print("\nF-Beta: %.4f\n" % f_beta)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=1),
    ModelCheckpoint(
        './xelibrion_weights_tf-%d.h5',
        monitor='val_loss',
        save_best_only=True),
    LearningRateScheduler(lr_scheduler),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1, ),
    Fbeta()
]

print("Starting training")
model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_val, Y_val),
    batch_size=model_batch_size,
    verbose=1,
    epochs=60,
    callbacks=callbacks,
    shuffle=True)
print("Training complete\n")
