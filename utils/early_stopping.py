import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, mode='max', delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = None
        self.counter = 0
        self.early_stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            self.counter = 0
            return False
        if self.mode == 'max':
            improved = metric > self.best + self.delta
        else:
            improved = metric < self.best - self.delta
        if improved:
            self.best = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
