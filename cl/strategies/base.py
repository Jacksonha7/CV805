"""Base class for all CL strategies."""
import torch
import numpy as np
from ..fusion_head import (FusionHead, ce_oracle_loss, soft_topk_loss,
                           build_optimizer, decode_lambda)


class BaseStrategy:
    """Train on one task; subclasses add regularisation or replay."""

    def __init__(self, model: FusionHead, lr: float = 1e-3,
                 loss_type: str = "ce_oracle", device: str = "cpu",
                 tau: float = 0.1, decode: str = "argmax"):
        self.model = model
        self.decode = decode
        self.lr = lr
        self.loss_type = loss_type
        self.tau = tau
        self.device = device
        self.optimizer = build_optimizer(model, lr=lr)
        # Set once by train_cl.py. Curves are addressed by integer id rather
        # than kept parallel to X/y, so replay buffers and joint concatenation
        # cannot silently mis-align a sample with another sample's curves.
        self.curve_bank = None
        self.lambda_grid = None

    def compute_loss(self, logits, labels, curve_ids=None):
        if self.loss_type == "soft_topk" and self.curve_bank is not None \
                and curve_ids is not None:
            curves = [self.curve_bank[int(c)] for c in curve_ids]
            return soft_topk_loss(logits, curves, self.lambda_grid,
                                  temperature=self.tau)
        return ce_oracle_loss(logits, labels)

    def train_task(self, train_X: np.ndarray, train_y: np.ndarray,
                   epochs: int = 50, batch_size: int = 64,
                   curve_ids: np.ndarray = None):
        self.model.train()
        X_t = torch.tensor(train_X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(train_y, dtype=torch.long, device=self.device)
        cid = None if curve_ids is None else np.asarray(curve_ids)

        n = len(X_t)
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(X_t[idx])
                cid_batch = None if cid is None else cid[idx.cpu().numpy()]
                loss = self.compute_loss(logits, y_t[idx], cid_batch)
                loss = self._add_regularisation(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(idx)

        self._after_task(X_t, y_t)

    def _add_regularisation(self, loss):
        return loss

    def _after_task(self, X, y):
        pass

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        logits = self.model(X_t)
        preds = logits.argmax(dim=-1)
        acc = (preds == y_t).float().mean().item()
        # "predictions" stays the argmax bin (λ-accuracy is a classification
        # metric); "lambdas" is what fusion actually uses.
        lambdas = decode_lambda(logits, self.decode)
        return {"accuracy": acc, "predictions": preds.cpu().numpy(),
                "lambdas": lambdas.cpu().numpy().astype(np.float64)}
