import sys
import os
import time
import cdt
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dag_gnn_utils2 import *

from torch.utils.data import DataLoader

# sys.path.append("..")
# from gran_dag.plot import plot_adjacency
# from gran_dag.utils.save import dump
# from gran_dag.utils.metrics import edge_errors
# from gran_dag.dag_optim import is_acyclic
# from gran_dag.data import DataManagerFile
# from gran_dag.train import cam_pruning_, pns_
# from dag_gnn.train import dag_gnn, retrain
# from dag_gnn.utils import load_numpy_data
from cdl.utils import is_dag

_EPS = 1e-10


class MLPEncoder(nn.Module):
    """MLP encoder module."""

    def __init__(self, n_xdims, hidden_layers, n_out, adj_A, mask_A, batch_size, do_prob=0., factor=True, tol=0.1):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # TODO: add masking for PNS
        self.mask_A = torch.Tensor(mask_A)
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)

        self.layers = nn.ModuleList([nn.Linear(n_xdims, hidden_layers[0], bias=True)])
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1], bias=True))
        self.layers.append(nn.Linear(hidden_layers[-1], n_out, bias=True))

        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):

        # to amplify the value of A and accelerate convergence.
        self.mask_A = self.mask_A.cuda()
        adj_A1 = torch.sinh(3. * self.adj_A * self.mask_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0], device='cuda').double()

        x = inputs
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        x = self.layers[-1](x)
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa


class MLPDecoder(nn.Module):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, adj_A_tilt, Wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa

        x = input_z
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        out = self.layers[-1](x)

        return mat_z, out, adj_A_tilt


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


prox_plus = torch.nn.Threshold(0., 0.)


def stau(w, tau):
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def is_acyclic(adjacency):
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0: return False
    return True


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

    return optimizer, lr


device = "cuda"


def dag_gnn(X,
            encoder_hidden_layers,
            decoder_hidden_layers,
            epochs=200,
            batch_size=100,
            lr=3e-3,
            lr_decay=200,
            gamma=1.0,
            encoder_dropout=0.0,
            decoder_dropout=0.0,
            graph_threshold = 0.3,
            tau_A=0.0,
            seed=1,
            ):
    torch.manual_seed(seed)
    np.random.seed(seed)

    LR = lr
    d = X.shape[1]

    train_loader = DataLoader(X, batch_size=batch_size, shuffle=False)  # todo

    # if opt.pns:
    #     initial_adj = pns_(initial_adj, train_data, test_data, opt.num_neighbors, opt.pns_thresh)

    # apply DAG-GNN
    off_diag = np.ones((d, d)) - np.eye(d)


    adj_A = np.zeros((d, d))
    mask_A = off_diag.copy()

    encoder = MLPEncoder(1, encoder_hidden_layers, 1, adj_A,
                         batch_size=batch_size,
                         do_prob=encoder_dropout, factor=True, mask_A=mask_A).double()

    decoder = MLPDecoder(1, 1,
                         batch_size=batch_size,
                         hidden_layers=decoder_hidden_layers).double()

    # adj_A = np.zeros((d, d))

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay,
                                          gamma=gamma)

    if device == "cuda":
        encoder.cuda()
        decoder.cuda()

    def train(train_loader, epoch, best_val_loss, best_graph, ground_truth_G, lambda_A, c_A, optimizer):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        encoder.train()
        decoder.train()
        scheduler.step()
        # print(rel_send)
        # update optimizer
        optimizer, lr = update_optimizer(optimizer, LR, c_A)

        for batch_idx, data in enumerate(train_loader):

            if device == "cuda":
                data = data.cuda()
            data = data.unsqueeze(2)
            data = Variable(data).double()

            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(
                data)  # logits is of size: [num_sims, z_dims]
            edges = logits

            dec_x, output, adj_A_tilt_decoder = decoder(edges, origin_A, adj_A_tilt_encoder, Wa)

            target = data
            preds = output
            variance = 0.

            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = tau_A * torch.sum(torch.abs(one_adj_A))

            # compute h(A)
            h_A = _h_A(origin_A, d)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(
                origin_A * origin_A) + sparse_loss  # +  0.01 * torch.sum(variance * variance)

            loss.backward()
            optimizer.step()

            myA.data = stau(myA.data, tau_A * lr)

            # compute metrics
            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < graph_threshold] = 0

            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            shd_trian.append(shd)

        if (np.mean(kl_train) + np.mean(nll_train)) < best_val_loss:
            best_graph = np.copy(graph)

        print(h_A.item())
        nll_val = []
        acc_val = []
        kl_val = []
        mse_val = []

        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train) + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'time: {:.4f}s'.format(time.time() - t))

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(
            mse_train), graph, origin_A, best_graph

    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []

    best_graph = np.ones((d, d)) - np.eye(d)
    # optimizer step on hyparameters
    c_A = 1
    lambda_A = 0.0
    h_A_new = torch.tensor(1.)
    h_tol = 1e-8
    k_max_iter = int(1e2)
    h_A_old = np.inf

    try:
        flag_max_iter = True
        for step_k in range(k_max_iter):
            while c_A < 1e+20:
                for epoch in range(epochs):
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A, best_graph = train(train_loader, epoch,
                                                                                       best_ELBO_loss, best_graph,
                                                                                       G, lambda_A, c_A,
                                                                                       optimizer)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                print("Optimization Finished!")
                print("Best Epoch: {:04d}".format(best_epoch))
                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, d)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A *= 10
                else:
                    break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                flag_max_iter = False
                break

        # return best_graph, rel_rec, rel_send, encoder, decoder, train_loader, flag_max_iter

    except KeyboardInterrupt:
        pass
        # print the best anway
        # print(best_ELBO_graph)
        # print(nx.to_numpy_array(ground_truth_G))
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
        # print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        #
        # print(best_NLL_graph)
        # print(nx.to_numpy_array(ground_truth_G))
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
        # print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        #
        # print(best_MSE_graph)
        # print(nx.to_numpy_array(ground_truth_G))
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
        # print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    w_adj = best_graph

    adj = (w_adj != 0).astype(np.double)

    # verify the masking worked properly
    # assert (initial_adj >= adj).all() # assert that edge_init = 0 => edge = 0

    # make sure is acyclic
    original_adj_cyclic = not is_acyclic(adj)
    while not is_acyclic(adj):
        print("Removing an edge since original DAG was not acyclic")
        w_adj_abs = np.abs(w_adj)
        min_abs_value = np.min(w_adj_abs[np.nonzero(w_adj_abs)])
        to_keep = (w_adj_abs > min_abs_value).astype(np.double)
        w_adj = w_adj * to_keep
        adj = (w_adj != 0).astype(np.double)

    # cam_pruning?
    # if opt.cam_pruning:
    #     new_adj = cam_pruning_(adj, train_data, test_data, opt.cutoff, opt.exp_path)
    #     assert (adj >= new_adj).all() # assert that cam_pruning is not adding edges
    #     adj = new_adj

    # evaluate held-out likelihood
    # if test_data.dataset is not None:
    #     test_data_np = test_data.dataset.unsqueeze(2).detach().cpu().numpy()
    # else:
    #     test_data_np = None
    # score_train, score_valid, flag_max_iter_retrain = retrain(opt, train_data_np, test_data_np, gt_dag, adj)

    # # Compute SHD and SID metrics
    # sid = float(cdt.metrics.SID(target=gt_dag, pred=adj))
    # shd = float(cdt.metrics.SHD(target=gt_dag, pred=adj, double_for_anticausal=False))
    # shd_cpdag = float(cdt.metrics.SHD_CPDAG(target=gt_dag, pred=adj))
    # fn, fp, rev = edge_errors(adj, gt_dag)
    # timing = time.time() - time0
    #
    # #save
    # if not os.path.exists(opt.exp_path):
    #     os.makedirs(opt.exp_path)
    #
    # metrics_callback(stage="dag_gnn", step=0,
    #                  metrics={"train_score": score_train, "val_score": score_valid, "sid": sid, "shd": shd,
    #                           "shd_cpdag": shd_cpdag, "fn": fn, "fp": fp, "rev": rev,
    #                           "original_adj_cyclic": original_adj_cyclic, "flag_max_iter": flag_max_iter,
    #                           "flag_max_iter_retrain": flag_max_iter_retrain},
    #                  throttle=False)
    #
    # dump(opt, opt.exp_path, 'opt')
    # dump(timing, opt.exp_path, 'timing', True)
    # dump(score_train, opt.exp_path, 'train_score', True)
    # dump(score_valid, opt.exp_path, 'test_score', True)
    # dump(sid, opt.exp_path, 'sid', True)
    # dump(shd, opt.exp_path, 'shd', True)
    # np.save(os.path.join(opt.exp_path, "DAG"), adj)

    # plot_adjacency(gt_dag, adj, opt.exp_path)


def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)


# todo add pns? cam pruning?
# todo: needed batch/sample size, optimizer, graph threshold, --tau_A, lambda_A, c_A, (A-connect, A-positiver loss), epochs, lr
# todo: normalize?
if __name__ == "__main__":
    from cdl.data.sachs import sachs

    G, X = sachs(as_df=False)
    encoder_hid_layers = [64]
    decoder_hid_layers = [64]
    dag_gnn(X, encoder_hidden_layers=encoder_hid_layers, decoder_hidden_layers=decoder_hid_layers)
