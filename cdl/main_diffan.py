from cdl.data.sachs import sachs
from cdl.models.diffan_old import DiffAN
from cdl.metrics import evaluate_graph
from cdl.utils import is_dag
from cdl.models.postprocessing.postprocessing import cam_pruning

G, X = sachs(as_df=False)

n_nodes = len(G)
diffan = DiffAN(n_nodes, residue= True, masking=False)
# G_est, order = diffan.fit(X)
G_est = diffan.fit(X)
T = cam_pruning(G_est, X, 0.001)
print(is_dag(G_est))
acc = evaluate_graph(G, G_est != 0)
print(acc)

print("")