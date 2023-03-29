from cdl.data.sachs import sachs
from torch.utils.data import DataLoader
from cdl.models.notears.nonlinear_lightning import NotearsMLP
import numpy as np
from cdl.utils import is_dag
from cdl.metrics import evaluate_graph
import pytorch_lightning as pl


G, dataset = sachs(as_df=False)
dataset = dataset.astype(np.float32)
train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

model = NotearsMLP([11, 1], max_iterations=100, lambda1=1, lambda2=1, w_threshold=0.3)
model.fit(train_loader, G, gpus=0)

W_est = model.fc1_to_adj()
W = abs(W_est) >= 0.3

print(f"Is dag: {is_dag(W)}")
acc = evaluate_graph(G, W)
print(acc)
print("")
