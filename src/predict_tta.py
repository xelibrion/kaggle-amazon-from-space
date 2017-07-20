#!/usr/bin/env python

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision.transforms as transforms
from train_pytorch import create_model, labels
from imgaug import augmenters as iaa


def define_args():
    parser = argparse.ArgumentParser(description='Kaggle Amazon from Space')

    parser.add_argument(
        '--checkpoint-file',
        default='model_best.pth.tar',
        type=str,
        metavar='N',
        help='path to model checkpoint file')

    parser.add_argument(
        '--images-dir',
        default='../input/test-jpg/',
        type=str,
        metavar='N',
        help='path to directory with images')

    parser.add_argument(
        '--use-gpu',
        default=False,
        action='store_true',
        help='flag indicates if we need to predict on GPU (default: false)')

    parser.add_argument(
        '-b',
        '--batch-size',
        default=256,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256)')

    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)')

    return parser


class LoadImageAsPIL(object):
    def __call__(self, img_path):
        image = Image.open(img_path)
        if image.mode == 'CMYK':
            image = image.convert('RGB')
        return image


class LoadImageAsNumpyArray(object):
    def __call__(self, img_path):
        return cv2.imread(img_path, -1)


class CenterCrop(object):
    def __init__(self):
        self.transform = iaa.Crop(px=(16, 16, 16, 16), keep_size=False)

    def __call__(self, img):
        return self.transform.augment_image(img)


class TopLeftCrop(object):
    def __init__(self):
        self.transform = iaa.Crop(px=(0, 32, 32, 0), keep_size=False)

    def __call__(self, img):
        return self.transform.augment_image(img)


class TopRightCrop(object):
    def __init__(self):
        self.transform = iaa.Crop(px=(0, 0, 32, 32), keep_size=False)

    def __call__(self, img):
        return self.transform.augment_image(img)


class BottomLeftCrop(object):
    def __init__(self):
        self.transform = iaa.Crop(px=(32, 32, 0, 0), keep_size=False)

    def __call__(self, img):
        return self.transform.augment_image(img)


class BottomRightCrop(object):
    def __init__(self):
        self.transform = iaa.Crop(px=(32, 0, 0, 32), keep_size=False)

    def __call__(self, img):
        return self.transform.augment_image(img)


class TTADataset(Dataset):
    def __init__(self, x_set, root_dir, transform_sets):
        self.x = x_set
        self.root_dir = root_dir
        self.transform_sets = transform_sets
        assert len(transform_sets) > 0

    def __len__(self):
        return len(self.x) * len(self.transform_sets)

    def __getitem__(self, idx):
        idx_x = idx // len(self.transform_sets) + 1
        img_name = self.x[idx_x]

        img_path = os.path.join(self.root_dir, "{}.jpg".format(img_name))

        return [t_set(img_path) for t_set in self.transform_sets]


def get_mlb():
    mlb = MultiLabelBinarizer(labels)
    dtype = np.int if all(isinstance(c, int) for c in mlb.classes) else object
    mlb.classes_ = np.empty(len(mlb.classes), dtype=dtype)
    mlb.classes_[:] = mlb.classes
    return mlb


def batch_to_labels(mlb, pred_batch):
    labels_tuples = mlb.inverse_transform(pred_batch)
    labels_str = [' '.join(x) for x in labels_tuples]
    return labels_str


def main():
    parser = define_args()
    args = parser.parse_args()

    model, _ = create_model(17)
    model = model.cuda()

    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    normalize = transforms.Normalize((0.302751, 0.344464, 0.315358),
                                     (0.127995, 0.132469, 0.152108))

    transform_sets = {
        'scaled':
        transforms.Compose([
            LoadImageAsPIL(),
            transforms.Scale(224),
            transforms.ToTensor(),
            normalize,
        ]),
        'center-crop':
        transforms.Compose([
            LoadImageAsNumpyArray(),
            CenterCrop(),
            transforms.ToTensor(),
            normalize,
        ]),
        'top-left-crop':
        transforms.Compose([
            LoadImageAsNumpyArray(),
            TopLeftCrop(),
            transforms.ToTensor(),
            normalize,
        ]),
        'top-right-crop':
        transforms.Compose([
            LoadImageAsNumpyArray(),
            TopRightCrop(),
            transforms.ToTensor(),
            normalize,
        ]),
        'bottom-left-crop':
        transforms.Compose([
            LoadImageAsNumpyArray(),
            BottomLeftCrop(),
            transforms.ToTensor(),
            normalize,
        ]),
        'bottom-right-crop':
        transforms.Compose([
            LoadImageAsNumpyArray(),
            BottomRightCrop(),
            transforms.ToTensor(),
            normalize,
        ]),
    }

    df_test = pd.read_csv('../input/sample_submission_v2.csv')

    threshold = 0.5

    result = []

    mlb = get_mlb()

    for image_name in tqdm(df_test['image_name'].values):
        img_path = os.path.join(args.images_dir, '%s.jpg' % image_name)

        tensors_seq = map(lambda x: x(img_path), transform_sets.values())
        tensors_seq = map(lambda x: x.expand([1, 3, 224, 224]), tensors_seq)
        input_t = torch.cat(tensors_seq)

        input_t = input_t.cuda(async=False)
        input_var = torch.autograd.Variable(input_t)
        out_t = model(input_var).float()

        y_pred = torch.sigmoid(out_t)
        y_pred = torch.ge(y_pred.mean(dim=0), threshold).float()
        pred_batch = y_pred.data.cpu().numpy()

        result.extend(batch_to_labels(mlb, np.expand_dims(pred_batch, 0)))

    df_test['tags'] = result
    df_test.to_csv('./my_sub_resnet50.csv', index=False)


if __name__ == '__main__':
    main()
