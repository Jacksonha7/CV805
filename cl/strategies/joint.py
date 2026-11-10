"""Joint training: train on ALL tasks at once (upper bound on accuracy)."""
import numpy as np
from .base import BaseStrategy


class JointStrategy(BaseStrategy):
    """Accumulates all seen data and retrains from scratch each round."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_X = None
        self.all_y = None
        self.all_cid = None

    def train_task(self, train_X, train_y, epochs=50, batch_size=64,
                   curve_ids=None):
        if self.all_X is None:
            self.all_X = train_X.copy()
            self.all_y = train_y.copy()
            self.all_cid = None if curve_ids is None else np.asarray(curve_ids).copy()
        else:
            self.all_X = np.concatenate([self.all_X, train_X])
            self.all_y = np.concatenate([self.all_y, train_y])
            if curve_ids is not None and self.all_cid is not None:
                self.all_cid = np.concatenate([self.all_cid, np.asarray(curve_ids)])

        # Reset model weights before training
        for layer in self.model.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.optimizer = type(self.optimizer)(
            self.model.parameters(), lr=self.lr)

        super().train_task(self.all_X, self.all_y, epochs, batch_size,
                           self.all_cid)
