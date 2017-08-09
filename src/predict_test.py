#!/usr/bin/env python

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm

import torchvision.transforms as transforms
from model_tuner import as_variable
from train import KaggleAmazonDataset, ParseNumFolds, create_model

DATA_ROOT = '../input'


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
        '-b',
        '--batch-size',
        default=72,
        type=int,
        metavar='N',
        help='mini-batch size (default: 72)')

    return parser


def jpg_absolute_path(image_name):
    return os.path.abspath(
        join(DATA_ROOT, 'test-jpg', '{}.jpg'.format(image_name)))


def create_data_pipeline(num_classes, args):
    df = pd.read_csv('../input/sample_submission_v2.csv')
    X_test = df['image_name'].apply(lambda x: jpg_absolute_path(x))

    normalize = transforms.Normalize([0.302751, 0.344464, 0.315358],
                                     [0.127995, 0.132469, 0.152108])

    test_loader = torch.utils.data.DataLoader(
        KaggleAmazonDataset(
            X_test,
            np.zeros((X_test.values.shape[0], num_classes)),
            transform=transforms.Compose([
                transforms.Scale(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available())

    return test_loader


def main():
    parser = define_args()
    args = parser.parse_args()

    num_classes = 17
    cudnn.benchmark = True

    print('Predicting test data using model for folds: {}'.format(args.folds))

    for fold in args.folds:

        model, _, _ = create_model(num_classes, 0)

        if torch.cuda.is_available():
            model = model.cuda()

        checkpoint_file = './model_best_fold_{}.pth.tar'.format(fold)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()

        test_loader = create_data_pipeline(num_classes, args)
        tq = tqdm(total=len(test_loader) * test_loader.batch_size)

        predict_batches = []

        for i, (inputs, target) in enumerate(test_loader):

            input_var = as_variable(inputs, volatile=True)

            output = model(input_var)
            prob_output = torch.sigmoid(output)
            predict_batches.append(prob_output.data.cpu().numpy())

            tq.update(test_loader.batch_size)

        tq.close()

        result = np.vstack(predict_batches)
        np.save('./test_predict_fold_{}.npy'.format(fold), result)


if __name__ == '__main__':
    main()
