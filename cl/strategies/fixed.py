"""Fixed λ baseline: always predict a constant λ (no learning)."""
import numpy as np
from .base import BaseStrategy
from ..oracle import LAMBDA_GRID


class FixedLambdaStrategy(BaseStrategy):
    """Always predict a fixed λ bin — serves as the training-free baseline."""

    def __init__(self, *args, fixed_lambda: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_bin = int(round(fixed_lambda * 10))

    def train_task(self, *args, **kwargs):
        pass  # no training

    def evaluate(self, X, y) -> dict:
        preds = np.full(len(y), self.fixed_bin, dtype=np.int64)
        acc = (preds == y).mean()
        return {"accuracy": float(acc), "predictions": preds,
                "lambdas": LAMBDA_GRID[preds]}


class OracleStrategy(BaseStrategy):
    """Always predict the oracle λ* — upper bound on accuracy."""

    def train_task(self, *args, **kwargs):
        pass

    def evaluate(self, X, y) -> dict:
        return {"accuracy": 1.0, "predictions": y.copy(),
                "lambdas": LAMBDA_GRID[y]}
