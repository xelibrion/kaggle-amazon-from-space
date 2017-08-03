import collections
import json
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

THRESHOLDS = np.array([
    0.185, 0.148, 0.164, 0.314, 0.193, 0.156, 0.05, 0.101, 0.192, 0.186, 0.169,
    0.213, 0.211, 0.183, 0.141, 0.289, 0.175
], 'float32')


def fbeta_score(y_true, y_pred, beta=2, threshold=0.2, eps=1e-9):
    beta_sq = beta**2

    threshold = torch.from_numpy(THRESHOLDS)
    if torch.cuda.is_available():
        threshold = threshold.cuda()

    y_pred = torch.ge(torch.sigmoid(y_pred.float()), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision *
         recall).div(precision.mul(beta_sq) + recall + eps).mul(1 + beta_sq))


def as_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda(async=True)
    return torch.autograd.Variable(tensor, volatile=volatile)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        self.reset(window_size)

    def reset(self, window_size):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window = collections.deque([], window_size)

    @property
    def mavg(self):
        return np.mean(self.window)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.window.append(self.val)


class Emitter:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, event):
        with open(self.path, 'a') as out_file:
            event.update({'timestamp': datetime.utcnow().isoformat()})
            out_file.write(json.dumps(event))
            out_file.write('\n')


class Tuner:
    def __init__(self,
                 model,
                 criterion,
                 bootstrap_optimizer,
                 optimizer,
                 bootstrap_epochs=1,
                 epochs=200,
                 early_stopping=None,
                 tag=None):
        self.model = model
        self.criterion = criterion
        self.bootstrap_optimizer = bootstrap_optimizer
        self.optimizer = optimizer
        self.bootstrap_epochs = bootstrap_epochs
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.start_epoch = 0
        self.best_score = -float('Inf')
        self.tag = tag
        self.emit = Emitter('./logs/events.json' if not tag else
                            './logs/events_{}.json'.format(tag))

    def restore_checkpoint(self, checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))

        checkpoint = torch.load(checkpoint_file)
        self.start_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))

    def save_checkpoint(self, validation_score, epoch):
        checkpoint_filename = ('checkpoint.pth.tar' if not self.tag else
                               'checkpoint_{}.pth.tar'.format(self.tag))
        best_model_filename = ('model_best.pth.tar' if not self.tag else
                               'model_best_{}.pth.tar'.format(self.tag))

        is_best = validation_score > self.best_score
        self.best_score = max(validation_score, self.best_score)

        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_score': self.best_score,
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, checkpoint_filename)
        if is_best:
            shutil.copyfile(checkpoint_filename, best_model_filename)

    def run(self, train_loader, val_loader):
        self.bootstrap(train_loader, val_loader)
        self.train_nnet(train_loader, val_loader)

    def train_nnet(self, train_loader, val_loader):

        scheduler = ReduceLROnPlateau(
            self.optimizer,
            'max',
            threshold_mode='rel',
            threshold=0.002,
            patience=3,
            min_lr=1e-7,
            verbose=True, )

        for epoch in range(self.start_epoch, self.epochs):
            self.train_epoch(train_loader, self.optimizer, epoch, 'training',
                             'Epoch #{epoch}')

            val_score = self.validate(val_loader, epoch, 'validation',
                                      'Validating #{epoch}')

            scheduler.step(val_score)

            if self.early_stopping:
                if self.early_stopping.should_trigger(
                        epoch,
                        val_score, ):
                    break

            self.save_checkpoint(val_score, epoch)

    def bootstrap(self, train_loader, val_loader):
        if self.start_epoch:
            return

        for epoch in range(self.bootstrap_epochs):
            self.train_epoch(train_loader, self.bootstrap_optimizer, epoch,
                             'bootstrap', 'Bootstrapping #{epoch}')
            self.validate(val_loader, epoch, 'bootstrap-val',
                          'Validating #{epoch}')

    def train_epoch(self, train_loader, optimizer, epoch, stage, format_str):
        batch_time = AverageMeter()
        losses = AverageMeter()
        f2_meter = AverageMeter()

        self.model.train()

        tq = tqdm(total=len(train_loader) * train_loader.batch_size)
        description = format_str.format(**locals())
        tq.set_description('{:16}'.format(description))

        batch_idx = -1
        end = time.time()
        for i, (inputs, target) in enumerate(train_loader):
            batch_idx += 1

            input_var = as_variable(inputs)
            target_var = as_variable(target)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            f2_score = fbeta_score(target_var.data, output.data)

            batch_size = inputs.size(0)
            losses.update(loss.data[0], batch_size)
            f2_meter.update(f2_score, batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time.mavg),
                loss='{:.3f}'.format(losses.mavg),
                f_beta='{:.3f}'.format(f2_meter.mavg), )
            tq.update(batch_size)

            self.emit({
                'stage': stage,
                'epoch': epoch,
                'batch': batch_idx,
                'f2_score': f2_score,
                'loss': losses.val
            })

            end = time.time()

        tq.close()

    def validate(self, val_loader, epoch, stage, format_str):
        batch_time = AverageMeter()
        losses = AverageMeter()
        f2_meter = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        tq = tqdm(total=len(val_loader) * val_loader.batch_size)
        description = format_str.format(**locals())
        tq.set_description('{:16}'.format(description))

        batch_idx = -1
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            batch_idx += 1

            input_var = as_variable(inputs, volatile=True)
            target_var = as_variable(target, volatile=True)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            f2_score = fbeta_score(target_var.data, output.data)

            batch_size = inputs.size(0)
            losses.update(loss.data[0], batch_size)
            f2_meter.update(f2_score, batch_size)

            batch_time.update(time.time() - end)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time.mavg),
                loss='{:.3f}'.format(losses.mavg),
                f_beta='{:.3f}'.format(f2_meter.mavg), )
            tq.update(batch_size)

            self.emit({
                'stage': stage,
                'epoch': epoch,
                'batch': batch_idx,
                'f2_score': f2_score,
                'loss': losses.val
            })
            end = time.time()

        tq.close()

        print('Validation results (avg): f2 score = {:.3f}, loss = {:.3f}\n'.
              format(f2_meter.avg, losses.avg))
        return f2_meter.avg
