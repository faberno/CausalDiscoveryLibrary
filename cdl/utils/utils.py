import numpy as np
import pandas as pd
from typing import Union
import uuid
import os

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


def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    #output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')
    output = os.path.join(save_path, 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output
