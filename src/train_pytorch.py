#!/usr/bin/env python

import argparse
import os
import shutil
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import torchvision.models as models
import torchvision.transforms as transforms

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
    '--use-gpu',
    default=False,
    action='store_true',
    help='flag indicates if we need to train on GPU (default: false)')
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
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=1e-3,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')

best_prec1 = 0


def create_model(num_classes):
    # create model
    print("=> using pre-trained model resnet50")
    model = models.resnet50(pretrained=True)
    # model = torch.nn.DataParallel(model).cpu()

    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules
    # have requires_grad=True by default
    model.fc = nn.Sequential(
        nn.Linear(2048, num_classes),
        nn.Sigmoid(), )

    return model


class KaggleAmazonDataset(Dataset):
    def __init__(self, x_set, y_set, root_dir, transform=None):
        self.x = x_set
        self.y = y_set
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
        y_tensor = torch.from_numpy(np.array(self.y[idx], dtype='float32'))

        return x_tensor, y_tensor


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


def get_x_y():
    sample_size = int(os.environ.get('SAMPLE_SIZE', 0))

    df_train = pd.read_csv('../input/train_v2.csv')
    df_train.sort_values('image_name', inplace=True)
    y_train = encode_labels(df_train.copy())

    if sample_size:
        x_t, _, y_t, _ = train_test_split(
            df_train['image_name'].values,
            y_train.values,
            train_size=sample_size)
        return x_t, y_t

    return df_train['image_name'].values, y_train.values


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = create_model(17)
    if args.use_gpu:
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
    # define loss function (criterion) and optimizer

    criterion = (nn.MultiLabelSoftMarginLoss().cuda()
                 if args.use_gpu else nn.MultiLabelSoftMarginLoss().cpu())
    # criterion = nn.CrossEntropyLoss().cpu()
    optimizer = torch.optim.Adam(model.fc.parameters(), args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    X, Y = get_x_y()
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225], )

    train_loader = torch.utils.data.DataLoader(
        KaggleAmazonDataset(
            X_train,
            Y_train,
            root_dir=args.train_dir,
            transform=transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.use_gpu)

    val_loader = torch.utils.data.DataLoader(
        KaggleAmazonDataset(
            X_val,
            Y_val,
            root_dir=args.train_dir,
            transform=transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.use_gpu)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f2_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_gpu:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        f2_score = fbeta_score(target, output.data)
        f2_meter.update(f2_score, input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'F2-score {f2_score.val:.3f} ({f2_score.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      f2_score=f2_meter))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    f2_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.use_gpu:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        f2_score = fbeta_score(target, output.data)
        f2_meter.update(f2_score, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'F2-score {f2_score.val:.3f} ({f2_score.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      f2_score=f2_meter, ))

    print(' * F2-score {f2_score.avg:.3f}'.format(f2_score=f2_meter))

    return f2_meter.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def fbeta_score(y_true, y_pred, beta=2, threshold=0.5, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision *
         recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))


if __name__ == '__main__':
    main()
