"""kNN memory baseline: no training, nearest-neighbour voting on λ*."""
import numpy as np
from .base import BaseStrategy
from ..oracle import LAMBDA_GRID


class KNNStrategy(BaseStrategy):
    """Store all seen (feature, λ*) pairs, predict by k-NN vote."""

    def __init__(self, *args, k: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.memory_X = None
        self.memory_y = None

    def train_task(self, train_X, train_y, **kwargs):
        if self.memory_X is None:
            self.memory_X = train_X.copy()
            self.memory_y = train_y.copy()
        else:
            self.memory_X = np.concatenate([self.memory_X, train_X])
            self.memory_y = np.concatenate([self.memory_y, train_y])

    def evaluate(self, X, y) -> dict:
        if self.memory_X is None or len(self.memory_X) == 0:
            preds = np.zeros(len(y), dtype=np.int64)
        else:
            # L2 distance
            dists = np.linalg.norm(
                X[:, None, :] - self.memory_X[None, :, :], axis=2)
            k = min(self.k, len(self.memory_X))
            nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
            # Majority vote
            preds = np.array([
                np.bincount(self.memory_y[nn_idx[i]],
                            minlength=11).argmax()
                for i in range(len(X))
            ], dtype=np.int64)

        acc = (preds == y).mean()
        return {"accuracy": float(acc), "predictions": preds,
                "lambdas": LAMBDA_GRID[preds]}
