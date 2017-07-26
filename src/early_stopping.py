class EarlyStopping:
    def __init__(self,
                 mode='min',
                 patience=5,
                 threshold=1e-4,
                 threshold_mode='rel',
                 verbose=True):
        self.patience = patience

        self.mode_worse = None
        self.is_better = None
        self.epochs_passed = -1
        self.verbose = verbose
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')

    def should_trigger(self, epoch, metric):
        current = metric
        self.epochs_passed = epoch + 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            if self.verbose:
                self.print_summary()
            return True

        return False

    def print_summary(self):
        print('Early stopping has triggered after {num_epochs} epochs.\n'
              'Best score: {best_score:.3f}, observed at epoch #{best_epoch}'.
              format(
                  num_epochs=self.epochs_passed,
                  best_score=self.best,
                  best_epoch=self.epochs_passed - self.patience - 2, ))
