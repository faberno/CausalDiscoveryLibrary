import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm

def pns(X, adj_start=None, num_neighbors=None, thresh=0.75):
    """Preliminary neighborhood selection"""
    num_nodes = X.shape[1]

    if adj_start is None:
        adj_start = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)

    for node in range(num_nodes):
        x_other = np.copy(X)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(x_other, X[:, node])
        selected_reg = SelectFromModel(reg, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(np.float)

        adj_start[:, node] *= mask_selected

    return adj_start