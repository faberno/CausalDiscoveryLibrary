import numpy as np
import pandas as pd
from typing import Union


def topological_sorting(A: Union[np.ndarray, pd.DataFrame]) -> Union[list, None]:
    """Kahn's algorithm to compute the topological order of a graph, if it exists.
    :param A: Adjacency matrix
    :return: List of sorted nodes or None if it doesn't exist.
    """
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    elif not isinstance(A, np.ndarray):
        raise ValueError("Adjacency matrix should be an array or dataframe.")
    L = []
    incoming_count = np.count_nonzero(A, axis=0)
    S = list(np.flatnonzero(incoming_count == 0))
    while len(S) != 0:
        n = S.pop()
        L.append(n)
        for m in np.flatnonzero(A[n]):
            incoming_count[m] -= 1
            if incoming_count[m] == 0:
                S.append(m)

    if incoming_count.any():
        return None
    return L


def is_dag(A: Union[np.ndarray, pd.DataFrame]) -> bool:
    """Checks if a graph is a DAG by Kahn's algorithm.
    :param A: Adjacency matrix
    :return: True if A is a dag, otherwise False.
    """
    if topological_sorting(A) is None:
        return False
    return True
