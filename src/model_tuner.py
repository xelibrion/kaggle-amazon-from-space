import collections
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm


def fbeta_score(y_true, y_pred, beta=2, threshold=0.5, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(torch.sigmoid(y_pred.float()), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision *
         recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))


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


class Tuner:
    def __init__(self, model, criterion, bootstrap_optimizer, optimizer,
                 bootstrap_epochs, epochs, use_gpu, print_freq):
        self.model = model
        self.criterion = criterion
        self.bootstrap_optimizer = bootstrap_optimizer
        self.optimizer = optimizer
        self.bootstrap_epochs = bootstrap_epochs
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.print_freq = print_freq

    def run(self, train_loader, val_loader, start_epoch=0):
        self.bootstrap(train_loader)

        best_fbeta = np.Inf

        for epoch in range(start_epoch, self.epochs):
            # train for one epoch
            self.train_epoch(train_loader, self.optimizer,
                             'Epoch #{}'.format(epoch))

            # evaluate on validation set
            fbeta = self.validate(val_loader, 'Validating #{}'.format(epoch))

            # remember best prec@1 and save checkpoint
            is_best = fbeta > best_fbeta
            best_fbeta = max(fbeta, best_fbeta)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_fbeta': best_fbeta,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def bootstrap(self, train_loader):
        """Bootstraps top layer(s) of the model before starting training"""

        for epoch in range(self.bootstrap_epochs):
            self.train_epoch(train_loader, self.bootstrap_optimizer,
                             'Boostrapping')

    def train_epoch(self, train_loader, optimizer, stage):
        batch_time = AverageMeter()
        losses = AverageMeter()
        f2_meter = AverageMeter()

        self.model.train()

        tq = tqdm(total=len(train_loader) * train_loader.batch_size)
        tq.set_description('{:15}'.format(stage))

        end = time.time()
        for i, (inputs, target) in enumerate(train_loader):

            if self.use_gpu:
                inputs = inputs.cuda(async=True)
                target = target.cuda(async=True)

            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            f2_score = fbeta_score(target, output.data)

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

            end = time.time()

        tq.close()

    def validate(self, val_loader, stage):
        batch_time = AverageMeter()
        losses = AverageMeter()
        f2_meter = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        tq = tqdm(total=len(val_loader) * val_loader.batch_size)
        tq.set_description('{:15}'.format(stage))

        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            if self.use_gpu:
                inputs = inputs.cuda(async=True)
                target = target.cuda(async=True)

            input_var = torch.autograd.Variable(inputs, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            f2_score = fbeta_score(target, output.data)

            batch_size = inputs.size(0)
            losses.update(loss.data[0], batch_size)
            f2_meter.update(f2_score, batch_size)

            batch_time.update(time.time() - end)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time.mavg),
                loss='{:.3f}'.format(losses.mavg),
                f_beta='{:.3f}'.format(f2_meter.mavg), )
            tq.update(batch_size)

            end = time.time()

        tq.close()

        print('Validation results (avg): f2 score = {:.3f}, loss = {:.3f}\n'.
              format(f2_meter.avg, losses.avg))
        return f2_meter.avg
