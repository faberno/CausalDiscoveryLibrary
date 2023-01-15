import os
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
def sachs(obs: bool = True, as_df: bool = True) -> (pd.DataFrame, pd.DataFrame):
    """Protein-Signaling Network by Sachs et al.
    - 11 nodes
    - 853 (obs) / 7466 (syn) samples

    :param obs: If true, the purely observational dataset is loaded, if not, the observational+synthetic one.
    :param as_df: If true return pd.Dataframes, if not return np.array.
    :return: Adjacency matrix (11x11) and dataset (853/7466x11) as dataframe.
    """
    G = pd.read_csv(
        os.path.join(root, "sachs_graph.csv"),
        index_col=0
    )

    data_name = "obs" if obs else "syn"
    X = pd.read_csv(
        os.path.join(root, f"sachs_{data_name}_data.csv")
    )
    if as_df:
        return G, X
    return G.to_numpy(), X.to_numpy()
