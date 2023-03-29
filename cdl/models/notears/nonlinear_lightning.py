from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union
from cdl.utils import is_dag
from cdl.metrics import evaluate_graph

from .locally_connected import LocallyConnected
from .lbfgsb_scipy import LBFGSBScipy
from .lbfgsb_pytorch import LBFGSB
from .trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math

import pytorch_lightning as pl
from pytorch_lightning.loops import FitLoop


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


class NotearsFitLoop(FitLoop):
    def __init__(self):
        super().__init__()

    def run(self, *args: Any, **kwargs: Any):
        if self.skip:
            return self.on_skip()

        self.reset()

        self.on_run_start(*args, **kwargs)

        model = self.trainer.model
        for k in range(model.max_iterations):
            while model.rho < model.rho_max:
                self.on_advance_start(*args, **kwargs)
                # print(model.fc1_pos.weight.data)
                self.advance(*args, **kwargs)
                self.on_advance_end()
                self._restarting = True
                self.reset()
                with torch.no_grad():
                    h_new = model.h_func().item()
                if h_new > 0.25 * model.h:
                    model.rho *= 10
                else:
                    model.h = h_new
                    break
            model.alpha += model.rho * h_new
            if model.rho >= model.rho_max:
                break
        self._restarting = False

        output = self.on_run_end()
        return output


class NotearsMLP(pl.LightningModule):
    def __init__(self, dims: List[int],
                 bias: bool = True,
                 max_iterations: int = 100,
                 lambda1: float = 0.,
                 lambda2: float = 0.,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.automatic_optimization = False
        self.max_iterations = max_iterations
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.rho = 1.0
        self.alpha = 0.0
        self.h = np.inf

        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def fit(self, X, G, **kwargs):
        trainer = pl.Trainer(**kwargs, log_every_n_steps=1)
        trainer.fit_loop = NotearsFitLoop()
        trainer.G = G
        trainer.fit(self, X)

    def configure_optimizers(self) -> Any:
        optimizer = LBFGSBScipy(self.parameters())
        # optimizer = LBFGSB(self.parameters(), line_search_fn="strong_wolfe")
        return optimizer

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        def closure():
            opt.zero_grad()
            X_hat = self(batch)
            loss = squared_loss(X_hat, batch)
            h_val = self.h_func()
            penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
            l2_reg = 0.5 * self.lambda2 * self.l2_reg()
            l1_reg = self.lambda1 * self.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            self.log_dict({'h': h_val.item(),
                           'loss': loss.item(),
                           'primal_obj': primal_obj.item()})
            self.manual_backward(primal_obj)
            return primal_obj

        opt.step(closure=closure)

    def training_epoch_end(self, outputs):
        if hasattr(self.trainer, "G"):
            W_est = self.fc1_to_adj()
            W = abs(W_est) >= self.w_threshold

            self.log('Is_DAG', float(is_dag(W)))
            acc = evaluate_graph(self.trainer.G, W)
            acc = {key: float(value) for key, value in acc.items()}
            self.log_dict(acc)

    def _bounds(self):
        d = self.dims[0]
        # ub = torch.full((d, d, self.dims[1]), np.inf)
        # ub[torch.eye(d, dtype=torch.bool), :] = 0
        # ub = torch.full((d, d), np.inf)
        # ub[torch.eye(d, dtype=torch.bool)] = 0
        # lb = torch.zeros((d, d))
        # return torch.stack((lb, ub))
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W