"""Elastic Weight Consolidation (Kirkpatrick et al., 2017)."""
import copy
import torch
import numpy as np
from .base import BaseStrategy


class EWCStrategy(BaseStrategy):
    def __init__(self, *args, ewc_lambda: float = 400.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher: dict = {}
        self.old_params: dict = {}

    def _add_regularisation(self, loss):
        if not self.fisher:
            return loss
        reg = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                reg += (self.fisher[name] *
                        (param - self.old_params[name]).pow(2)).sum()
        return loss + 0.5 * self.ewc_lambda * reg

    def _after_task(self, X, y):
        """Compute Fisher information on the task just finished."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters()}

        logits = self.model(X)
        for i in range(len(X)):
            self.model.zero_grad()
            log_prob = torch.nn.functional.log_softmax(logits[i], dim=0)
            log_prob[y[i]].backward(retain_graph=(i < len(X) - 1))
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

        for n in fisher:
            fisher[n] /= len(X)
            if n in self.fisher:
                self.fisher[n] = self.fisher[n] + fisher[n]
            else:
                self.fisher[n] = fisher[n]

        self.old_params = {n: p.detach().clone()
                           for n, p in self.model.named_parameters()}
