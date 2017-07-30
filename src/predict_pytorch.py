#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision.transforms as transforms
from train_pytorch import create_model, labels


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


class PredictDataset(Dataset):
    def __init__(self, x_set, root_dir, transform=None):
        self.x = x_set
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "{}.jpg".format(self.x[idx]))
        image = Image.open(img_name)
        if image.mode == 'CMYK':
            image = image.convert('RGB')

        x_tensor = self.transform(image) if self.transform else image

        return x_tensor


def main():
    parser = define_args()
    args = parser.parse_args()

    model, _ = create_model(17)
    model = model.cuda()

    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    normalize = transforms.Normalize([0.302751, 0.344464, 0.315358],
                                     [0.127995, 0.132469, 0.152108])

    apply_aug = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        normalize,
    ])

    df_test = pd.read_csv('../input/sample_submission_v2.csv')

    predict_loader = torch.utils.data.DataLoader(
        PredictDataset(
            df_test['image_name'].values,
            root_dir=args.images_dir,
            transform=apply_aug),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    t = tqdm(total=df_test.shape[0])

    threshold = 0.5

    mlb = MultiLabelBinarizer(labels)
    dtype = np.int if all(isinstance(c, int) for c in mlb.classes) else object
    mlb.classes_ = np.empty(len(mlb.classes), dtype=dtype)
    mlb.classes_[:] = mlb.classes

    result = []

    for i, input_t in enumerate(predict_loader):
        input_t = input_t.cuda(async=True)

        input_var = torch.autograd.Variable(input_t)

        y_pred = model(input_var)
        y_pred = torch.ge(torch.sigmoid(y_pred.float()), threshold).float()
        pred_batch = y_pred.data.cpu().numpy()

        labels_tuples = mlb.inverse_transform(pred_batch)
        labels_str = [' '.join(x) for x in labels_tuples]
        result.extend(labels_str)

        t.update(min(pred_batch.shape[0], args.batch_size))

    df_test['tags'] = result
    df_test.to_csv('./my_sub_resnet50.csv', index=False)


if __name__ == '__main__':
    main()
