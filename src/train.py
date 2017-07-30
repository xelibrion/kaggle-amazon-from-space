#!/usr/bin/env python

import argparse
import os
import random

import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from model_tuner import Tuner
import augmentations
from early_stopping import EarlyStopping
from torchvision import models


class ParseNumFolds(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ParseNumFolds, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if '-' in values:
            bounds = values.split('-')
            assert len(bounds) == 2
            namespace.folds = list(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            namespace.folds = [int(values)]


def define_args():

    parser = argparse.ArgumentParser(description='Kaggle Amazon from Space')

    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)')

    parser.add_argument(
        '--train-dir',
        default='../input/train-jpg',
        type=str,
        metavar='N',
        help='path to the folder containing images'
        'from train set (default: ../input/train-jpg/)')

    parser.add_argument(
        '--folds',
        default=[1],
        action=ParseNumFolds,
        metavar='N or M-N',
        help='fold to train on (default: 1)')

    parser.add_argument(
        '--epochs',
        default=90,
        type=int,
        metavar='N',
        help='number of total epochs to run')

    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)')

    parser.add_argument(
        '--epoch-size',
        default=None,
        type=int,
        metavar='N',
        help='manual epoch size (useful for debugging)')

    parser.add_argument(
        '-b',
        '--batch-size',
        default=72,
        type=int,
        metavar='N',
        help='mini-batch size (default: 72)')

    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=1e-2,
        type=float,
        metavar='LR',
        help='initial learning rate')

    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)')

    return parser


best_fbeta = 0


def create_model(num_classes, initial_lr):
    print("=> using pre-trained model resnet50")
    model = models.resnet50(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False

    model.fc = nn.Linear(2048, num_classes)

    optim_config = [{
        'params': model.layer1.parameters(),
        'lr': initial_lr * 1e-3
    }, {
        'params': model.layer2.parameters(),
        'lr': initial_lr * 1e-3
    }, {
        'params': model.layer3.parameters(),
        'lr': initial_lr * 1e-2
    }, {
        'params': model.layer4.parameters(),
        'lr': initial_lr * 1e-2
    }, {
        'params': model.fc.parameters()
    }]

    return model, model.fc.parameters(), optim_config


class KaggleAmazonDataset(Dataset):
    def __init__(self, x_set, y_set, transform=None, epoch_size=None):
        self.x = x_set
        self.y = y_set
        self.transform = transform
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size or len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        if image.mode == 'CMYK':
            image = image.convert('RGB')

        x_tensor = self.transform(image) if self.transform else image
        y_tensor = torch.from_numpy(np.array(self.y[idx], dtype='float32'))

        return x_tensor, y_tensor


def get_class_weights(labels, y):
    df_y = pd.DataFrame(y, columns=labels)
    df_w = 1 / (df_y.sum() / df_y.shape[0])
    df_w = df_w.apply(np.log) + 1
    weights_dict = df_w.to_dict()
    return [weights_dict[x] for x in labels]


class_weights = np.array([
    2.1899652895442587, 5.7825384927184471, 4.8492833294352042,
    5.803403631182408, 7.0235711214283247, 1.3533032219605925,
    3.9640978385423313, 7.0033684141108061, 3.2018301418579314,
    3.4033201737034853, 3.7086432769591671, 2.7182657607186504,
    1.0760957815487335, 2.6125059307745246, 5.7795929824886896,
    6.2662043481340861, 2.697817938147538
], 'float32')


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def create_data_pipeline(fold):
    print("Loading data")

    train_path = os.path.join('../input/fold_{}/train.csv'.format(fold))
    df_train = pd.read_csv(train_path)
    X_train = df_train['image_path'].values
    Y_train = df_train[df_train.columns.drop('image_path')].values

    val_path = os.path.join('../input/fold_{}/val.csv'.format(fold))
    df_val = pd.read_csv(val_path)
    X_val = df_val['image_path'].values
    Y_val = df_val[df_val.columns.drop('image_path')].values

    normalize = transforms.Normalize([0.302751, 0.344464, 0.315358],
                                     [0.127995, 0.132469, 0.152108])

    train_loader = torch.utils.data.DataLoader(
        KaggleAmazonDataset(
            X_train,
            Y_train,
            epoch_size=args.epoch_size,
            transform=transforms.Compose([
                transforms.RandomSizedCrop(224),
                augmentations.D4(),
                augmentations.Add(-5, 5, per_channel=True),
                augmentations.ContrastNormalization(
                    0.8,
                    1.2,
                    per_channel=True, ),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available())

    val_loader = torch.utils.data.DataLoader(
        KaggleAmazonDataset(
            X_val,
            Y_val,
            epoch_size=args.epoch_size,
            transform=transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available())

    return train_loader, val_loader


def main():
    global args, best_fbeta
    parser = define_args()
    args = parser.parse_args()

    num_classes = 17
    cudnn.benchmark = True

    print('Training model(s) for folds: {}'.format(args.folds))

    for fold in args.folds:

        model, bootstrap_params, full_params = create_model(
            num_classes,
            args.lr, )
        criterion = nn.MultiLabelSoftMarginLoss(
            size_average=False,
            weight=torch.from_numpy(class_weights), )

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()

        bootstrap_optimizer = torch.optim.Adam(bootstrap_params, args.lr)
        optimizer = torch.optim.Adam(full_params, args.lr)

        train_loader, val_loader = create_data_pipeline(fold)

        tuner = Tuner(
            model,
            criterion,
            bootstrap_optimizer,
            optimizer,
            tag='fold_{}'.format(fold),
            early_stopping=EarlyStopping(mode='max', patience=7))

        if args.resume:
            if os.path.isfile(args.resume):
                tuner.restore_checkpoint(args.resume)

        tuner.run(train_loader, val_loader)


if __name__ == '__main__':
    main()
