import torch

class ReducelrOnPlateau:
    def __init__(self, optimizer, patience = 2, factor = 0.5, min_lr = 1e-4, threshold = 1e-4):
        self.optimizer = optimizer
        self.threshold = threshold
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.best = float("inf")
        self.patience_counter = 0

    def step(self, loss):
        if loss is None:
            raise ValueError("Scheduler loss is None")

        if loss < self.best - self.threshold:
            self.best = loss
            self.patience_counter = 0

        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            print(f"Scheduler: Reducing lr to {new_lr}")
            self.optimizer.lr = new_lr
            self.patience_counter = 0

    def get_config(self):
        return {
            "type": "ReducelrOnPlateau",
            "patience": self.patience,
            "factor": self.factor,
            "min_lr": self.min_lr,
            "threshold": self.threshold
        }

