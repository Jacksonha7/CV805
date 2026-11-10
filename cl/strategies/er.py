"""Experience Replay: keep a fixed-size memory buffer of old samples."""
import numpy as np
import torch
from .base import BaseStrategy


class ERStrategy(BaseStrategy):
    def __init__(self, *args, buffer_size: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.buffer_X: np.ndarray = None
        self.buffer_y: np.ndarray = None
        self.buffer_cid: np.ndarray = None

    def train_task(self, train_X, train_y, epochs=50, batch_size=64,
                   curve_ids=None):
        if self.buffer_X is not None:
            combined_X = np.concatenate([train_X, self.buffer_X])
            combined_y = np.concatenate([train_y, self.buffer_y])
            combined_cid = None if curve_ids is None else \
                np.concatenate([np.asarray(curve_ids), self.buffer_cid])
        else:
            combined_X = train_X
            combined_y = train_y
            combined_cid = curve_ids

        super().train_task(combined_X, combined_y, epochs, batch_size,
                           combined_cid)

        self._update_buffer(train_X, train_y, curve_ids)

    def _update_buffer(self, new_X, new_y, new_cid=None):
        new_cid = np.full(len(new_X), -1) if new_cid is None \
            else np.asarray(new_cid)
        if self.buffer_X is None:
            self.buffer_X = new_X.copy()
            self.buffer_y = new_y.copy()
            self.buffer_cid = new_cid.copy()
        else:
            self.buffer_X = np.concatenate([self.buffer_X, new_X])
            self.buffer_y = np.concatenate([self.buffer_y, new_y])
            self.buffer_cid = np.concatenate([self.buffer_cid, new_cid])

        # Reservoir sampling to keep buffer_size. The curve ids are subsampled
        # with the SAME indices, otherwise a replayed feature vector would be
        # scored against a different query's score curves.
        if len(self.buffer_X) > self.buffer_size:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.buffer_X), self.buffer_size,
                             replace=False)
            self.buffer_X = self.buffer_X[idx]
            self.buffer_y = self.buffer_y[idx]
            self.buffer_cid = self.buffer_cid[idx]
