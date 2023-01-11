import os
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
def sachs(obs=True):
    """Protein-Signaling Network by Sachs et al.
    - 11 nodes
    - 853 (obs) / 7466 (syn) samples

    Parameters
    ----------
    obs : bool
        If true, the purely observational dataset is loaded, if not, the observational+synthetic one.
    """
    G = pd.read_csv(
        os.path.join(root, "sachs_graph.csv"),
        index_col=0
    )

    data_name = "obs" if obs else "syn"
    X = pd.read_csv(
        os.path.join(root, f"sachs_{data_name}_data.csv")
    )
    return G, X
