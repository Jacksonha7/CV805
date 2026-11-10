"""Learning without Forgetting (Li & Hoiem, 2017)."""
import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseStrategy


class LwFStrategy(BaseStrategy):
    def __init__(self, *args, lwf_alpha: float = 1.0,
                 lwf_temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lwf_alpha = lwf_alpha
        self.temperature = lwf_temperature
        self.old_logits: torch.Tensor = None
        self.task_X: torch.Tensor = None

    def train_task(self, train_X, train_y, epochs=50, batch_size=64,
                   curve_ids=None):
        X_t = torch.tensor(train_X, dtype=torch.float32, device=self.device)

        # Record soft targets from old model on new data
        if self.old_logits is not None:
            with torch.no_grad():
                self.model.eval()
                self._current_old_logits = self.model(X_t).detach()
        else:
            self._current_old_logits = None

        self.task_X = X_t
        super().train_task(train_X, train_y, epochs, batch_size, curve_ids)

    def _add_regularisation(self, loss):
        if self._current_old_logits is None or self.task_X is None:
            return loss
        # Distillation on full current-task data is expensive;
        # approximate on the current batch indices
        with torch.no_grad():
            old_soft = F.softmax(
                self._current_old_logits / self.temperature, dim=-1)
        new_logits = self.model(self.task_X)
        new_log = F.log_softmax(new_logits / self.temperature, dim=-1)
        dist_loss = F.kl_div(new_log, old_soft, reduction="batchmean")
        return loss + self.lwf_alpha * (self.temperature ** 2) * dist_loss

    def _after_task(self, X, y):
        with torch.no_grad():
            self.model.eval()
            self.old_logits = self.model(X).detach()
