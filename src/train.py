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


x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('../input/train_v2.csv')
df_test = pd.read_csv('../input/sample_submission_v2.csv')

labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = [
    'blow_down', 'bare_ground', 'conventional_mine', 'blooming', 'cultivation',
    'artisinal_mine', 'haze', 'primary', 'slash_burn', 'habitation', 'clear',
    'road', 'selective_logging', 'partly_cloudy', 'agriculture', 'water',
    'cloudy'
]

label_map = {
    'agriculture': 14,
    'artisinal_mine': 5,
    'bare_ground': 1,
    'blooming': 3,
    'blow_down': 0,
    'clear': 10,
    'cloudy': 16,
    'conventional_mine': 2,
    'cultivation': 4,
    'habitation': 9,
    'haze': 6,
    'partly_cloudy': 13,
    'primary': 7,
    'road': 11,
    'selective_logging': 12,
    'slash_burn': 8,
    'water': 15
}
num_classes = len(labels)
img_size = 224

# print("Resizing test set")
# for f, tags in tqdm(df_test.values, miniters=1000):
#     img = cv2.imread('../input/test-tif-v2/{}.tif'.format(f), -1)
#     x_test.append(cv2.resize(img, (img_size, img_size)))
# x_test = np.array(x_test, np.float32) / 255.

print("Resizing train set")
for f, tags in tqdm(df_train.values, miniters=1000):
    img_path = '{}/{}.jpg'.format(train_dir, f)
    if not os.path.exists(img_path):
        continue
    # https://stackoverflow.com/questions/37512119/resize-transparent-image-in-opencv-python-cv2
    # If you load a 4 channel image, the flag -1 indicates that the image
    # is loaded unchanged, so you can load and split all 4 channels directly.
    img = cv2.imread(img_path, -1)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (img_size, img_size)))
    y_train.append(targets)
y_train = np.array(y_train, np.uint8)
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
        histogram_freq=1,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None),
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
