from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union
import logging

import pytorch_lightning as pl
from pytorch_lightning.loops import FitLoop

from dag_gnn_utils import *

import numpy as np

from cdl.data.sachs import sachs


log = logging.getLogger(__name__)


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return lr

class CustomFitLoop(FitLoop):
    def __init__(self, k_max_iter=100, max_epochs=20):
        super().__init__()
        self.k_max_iter = k_max_iter
        self.max_epochs = max_epochs


    def run(self, *args: Any, **kwargs: Any):
        if self.skip:
            return self.on_skip()

        self.reset()

        self.max_epochs = self.trainer.max_epochs
        self.on_run_start(*args, **kwargs)

        model = self.trainer.model
        for k in range(self.k_max_iter):
            print("------------------------------------------------------------")
            print(f"--------------------Iteration {k}---------------------")
            print("------------------------------------------------------------")
            while model.current_c_A < 1e20:
                print(f"c_A: {model.current_c_A}")
                for i in range(self.max_epochs):
                    self.on_advance_start(*args, **kwargs)
                    self.advance(*args, **kwargs)
                    self.on_advance_end()
                    self._restarting = False
                self._restarting = True
                self.reset()

                new_h_A = _h_A(model.current_graph, model.d)
                if new_h_A.item() > 0.25 * model.old_h_A:
                    model.current_c_A *= 10
                else:
                    break
            model.old_h_A = new_h_A.item()
            model.current_lambda_A += model.current_c_A * new_h_A.item()

            if new_h_A.item() <= model.h_tol:
                break

        self._restarting = False

        output = self.on_run_end()
        return output

    def advance(self) -> None:
        """Runs one whole epoch."""
        log.detail(f"{self.__class__.__name__}: advancing loop")
        assert self.trainer.train_dataloader is not None
        dataloader = self.trainer.train_dataloader

        def batch_to_device(batch: Any) -> Any:
            batch = self.trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=0)
            batch = self.trainer._call_strategy_hook("batch_to_device", batch, dataloader_idx=0)
            return batch

        model = self.trainer.optimizers[0]
        lr = update_optimizer(self.trainer.optimizers[0], self.trainer.model.initial_lr, self.trainer.model.current_c_A)
        self.trainer.model.current_lr = lr

        assert self._data_fetcher is not None
        self._data_fetcher.setup(dataloader, batch_to_device=batch_to_device)
        with self.trainer.profiler.profile("run_training_epoch"):
            self._outputs = self.epoch_loop.run(self._data_fetcher)


class MLPEncoder(pl.LightningModule):
    """MLP encoder module."""

    def __init__(self, n_xdims, hidden_layers, n_out, adj_A, mask_A, batch_size, factor=True, tol=0.1):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(
            Variable(adj_A, requires_grad=True))  # TODO: add masking for PNS
        self.mask_A = mask_A
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)

        self.layers = nn.ModuleList([nn.Linear(n_xdims, hidden_layers[0], bias=True)])
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1], bias=True))
        self.layers.append(nn.Linear(hidden_layers[-1], n_out, bias=True))

        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(adj_A))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.manual_seed(1)
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        device = self.adj_A.get_device()
        self.mask_A = self.mask_A.to(device)
        self.Wa = self.Wa.to(device)
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3. * self.adj_A * self.mask_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0])

        x = inputs
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        x = self.layers[-1](x)
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return logits, adj_A1, adj_A


class MLPDecoder(pl.LightningModule):
    """MLP decoder module."""

    def __init__(self, n_in_z, n_out, batch_size, hidden_layers):
        super(MLPDecoder, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(n_in_z, hidden_layers[0], bias=True)])
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1], bias=True))
        self.layers.append(nn.Linear(hidden_layers[-1], n_out, bias=True))

        self.batch_size = batch_size

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.manual_seed(1)
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, adj_A_tilt, Wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa

        x = mat_z
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        out = self.layers[-1](x)

        return out


def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def stau(w, tau):
    w1 = F.relu(torch.abs(w) - tau)
    return torch.sign(w) * w1




class LitDagGNN(pl.LightningModule):
    def __init__(self,
                 d,
                 encoder_hidden_layers,
                 decoder_hidden_layers,
                 k_max_iterations = 100,
                 epochs=200,
                 batch_size=100,
                 lr=3e-3,
                 lr_decay=200,
                 gamma=1.0,
                 graph_threshold=0.3,
                 tau_A=0.0,
                 lambda_A = 0.0,
                 c_A = 1.0,
                 seed=None):
        super().__init__()
        self.d = d
        self.epochs = epochs
        self.k_max_iterations = k_max_iterations
        self.batch_size = batch_size
        self.initial_lr = lr
        self.current_lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.graph_threshold = graph_threshold
        self.tau_A = tau_A
        self.initial_lambda_A = lambda_A
        self.current_lambda_A = lambda_A
        self.initial_c_A = c_A
        self.current_c_A = c_A
        self.seed = seed

        self.current_graph = None
        self.old_h_A = torch.inf
        self.current_h_A = torch.tensor(1.0)
        self.h_tol = 1e-8
        self.current_iteration_k = 0


        if seed is not None:
            pl.seed_everything(seed)

        #todo add dropout

        adj_A = torch.zeros((d, d)) # todo make these inputs
        mask_A = torch.ones((d, d)) - torch.eye(d)

        self.encoder = MLPEncoder(1, encoder_hidden_layers, 1, adj_A,
                         batch_size=batch_size, factor=True, mask_A=mask_A)

        self.decoder = MLPDecoder(1, 1,
                         batch_size=batch_size,
                         hidden_layers=decoder_hidden_layers)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay, gamma=self.gamma)
        return [optimizer], [scheduler]

    def forward(self, x):
        edges, origin_A, adj_A_tilt_encoder = self.encoder(x)
        predictions = self.decoder(edges, origin_A, adj_A_tilt_encoder, self.encoder.Wa)

        return edges, origin_A, predictions


    def training_step(self, train_batch, batch_idx):
        targets = train_batch

        if len(targets.shape) == 2:
            targets = torch.unsqueeze(targets, 2)

        torch.manual_seed(1)
        edges, origin_A, predictions = self(targets)
        h_A = _h_A(origin_A, self.d)

        variance = 0.0
        nll_loss = nll_gaussian(predictions, targets, variance)
        kl_loss = kl_gaussian_sem(edges)
        sparse_loss = self.tau_A * torch.sum(torch.abs(origin_A))
        lagrangian_loss = self.current_lambda_A * h_A + 0.5 * self.current_c_A * h_A * h_A + 100. * torch.trace(
            origin_A * origin_A)

        loss = nll_loss + kl_loss + sparse_loss + lagrangian_loss # todo custom lr scheduler

        self.log('nll_loss', nll_loss.item())
        self.log('kl_loss', kl_loss.item())
        self.log('sparse_loss', sparse_loss.item())
        self.log('lagrangian_loss', lagrangian_loss.item())
        self.log('loss', loss.item())

        self.current_graph = origin_A

        return loss

    def training_epoch_end(self, outputs):
        G = self.trainer.train_dataloader.loaders.G

        graph_ = self.current_graph.detach().cpu().clone().numpy()
        graph_[np.abs(graph_) < self.graph_threshold] = 0
        fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(graph_))
        self.log('FDR', fdr)
        self.log('TPR', tpr)
        self.log('FPR', fpr)
        self.log('SHD', shd)
        self.log('NNZ', nnz)


    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        torch.manual_seed(1)
        loss.backward()

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        torch.manual_seed(1)
        optimizer.step(closure=optimizer_closure)
        self.encoder.adj_A.data = stau(self.encoder.adj_A.data, self.tau_A * self.current_lr)


G, dataset = sachs(as_df=False)
dataset = dataset.astype(np.float32)
train_loader = DataLoader(dataset, batch_size=100, shuffle=False)
train_loader.G = G

model = LitDagGNN(11, [64], [64])

trainer = pl.Trainer(gpus=1)
trainer.fit_loop = CustomFitLoop(k_max_iter=100, max_epochs=200)
trainer.fit(model, train_loader)
