import shutil
import time

import numpy as np
import torch


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

        epoch_time = AverageMeter()
        print("Starting training")
        for epoch in range(start_epoch, self.epochs):
            end = time.time()

            # train for one epoch
            self.train(self.optimizer, train_loader, epoch)

            # evaluate on validation set
            fbeta = self.validate(val_loader)

            # remember best prec@1 and save checkpoint
            is_best = fbeta > best_fbeta
            best_fbeta = max(fbeta, best_fbeta)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_fbeta': best_fbeta,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

            epoch_time.update(time.time() - end)
            print(
                ' * Time taken: {epoch_time.val:.1f}s ({epoch_time.avg:.1f}s)\n'.  # noqa
                format(epoch_time=epoch_time))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def bootstrap(self, train_loader):
        """Bootstraps top layer(s) of the model before starting training"""
        epoch_time = AverageMeter()

        print("Started bootstrapping")
        for epoch in range(self.bootstrap_epochs):
            end = time.time()

            self.train(self.bootstrap_optimizer, train_loader, epoch)

            epoch_time.update(time.time() - end)
            print(
                ' * Time taken: {epoch_time.val:.1f}s ({epoch_time.avg:.1f}s)\n'.  # noqa
                format(epoch_time=epoch_time))

        print("Boostraping finished\n")

    def train(self, optimizer, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        f2_meter = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if self.use_gpu:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

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

            if i % self.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
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

    def validate(self, val_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()
        f2_meter = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if self.use_gpu:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            losses.update(loss.data[0], input.size(0))
            f2_score = fbeta_score(target, output.data)
            f2_meter.update(f2_score, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
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
