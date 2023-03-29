import pandas as pd
import glob
import numpy as np
import torch


from cdl.data.sachs import sachs
from cdl.metrics import FPR, evaluate_graph
# from cdl.models.notears import notears_linear
import cdl.utils as utils
from cdl.models.notears import notears_linear, notears_nonlinear, NotearsMLP

np.random.seed(1)
torch.manual_seed(1)

from cdl.data.bnlearn import load_bnlearn_dataset

# a = load_bnlearn_dataset("asia", 0)
# vals = {'yes': 1, 'no':0}
# b = {key:[vals[v] for v in val] for (key, val) in a[2].items()}
# X = a[1].replace(vals)
# G = a[0].to_numpy()
#
G, X = sachs(as_df=False)
X = X.astype(np.float32)
# X = X.to_numpy().astype(np.float32)
# X = X + np.random.uniform(0, 0.5, size=X.shape).astype(np.float32)
d = len(G)

model = NotearsMLP(dims=[d,2,  1], bias=True)
W_est = notears_nonlinear(model, X, lambda1=1, lambda2=1, w_threshold=0.0)
W = abs(W_est) >= 0.3
print(f"------Normal------")
print(f"Is dag: {utils.is_dag(W)}")
acc = evaluate_graph(G, W)
print(acc)


# for i in [0.25, 0.5, 1, 2, 4, 8, 10, 15]:
#     X_ = (X - X.min(axis=0) / X.max(axis=0) - X.min(axis=0)) * i
#     model = NotearsMLP(dims=[d, 1], bias=True)
#     W_est = notears_nonlinear(model, X_, lambda1=1, lambda2=1, w_threshold=0.0)
#     W = abs(W_est) >= 0.2
#     print(f"------{i}------")
#     print(f"Is dag: {utils.is_dag(W)}")
#     acc = evaluate_graph(G, W)
#     print(acc)
# W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
# print("")
# model = NotearsMLP(dims=[d, 1], bias=True)
# W_est = notears_nonlinear(model, X, lambda1=1, lambda2=1, w_threshold=0.0)
# print("")
#
#
# W = abs(W_est) >= 0.2
# print(f"Is dag: {utils.is_dag(W)}")
# acc = evaluate_graph(G, W)
# print(acc)
