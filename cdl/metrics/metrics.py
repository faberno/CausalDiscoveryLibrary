import numpy as np


def SHD(G: np.array, E: np.array, reverse_double=False) -> int:
    """Compute the Structural Hamming Distance between two graphs (FP + FN + reversed).
    :param G: Ground-truth adjacency matrix
    :param E: Estimated adjacency Matrix
    :param reverse_double: a reversed edge should be counted as two false edges
    :return: SHD between G and E
    """
    if not ((E == 0) | (E == 1)).all() and ((G == 0) | (G == 1)).all():
        raise ValueError('The matrices should be binary.')
    if G.dtype != bool:
        G = G.astype(bool)
    if E.dtype != bool:
        E = E.astype(bool)
    if reverse_double:
        return np.count_nonzero(E != G)
    reversed = (E.T & G & (~E)).T  # edges that when transposed are the same as G but not as E anymore
    return np.count_nonzero((E ^ reversed) != G)  # remove the reversed edges once, so they don't count double


def TPR(G: np.array, E: np.array) -> float:
    """Compute the True Positive Rate between two graphs (TP / T).
    :param G: Ground-truth adjacency matrix
    :param E: Estimated adjacency Matrix
    :return: TPR between G and E
    """
    if not ((E == 0) | (E == 1)).all() and ((G == 0) | (G == 1)).all():
        raise ValueError('The matrices should be binary.')
    if G.dtype != bool:
        G = G.astype(bool)
    if E.dtype != bool:
        E = E.astype(bool)
    return np.count_nonzero(G & E) / max(np.count_nonzero(G), 1)


def FPR(G: np.array, E: np.array) -> float:
    """Compute the False Positive Rate between two graphs ((FP + reversed) / F)
    :param G: Ground-truth adjacency matrix
    :param E: Estimated adjacency Matrix
    :return: FPR between G and E
    """
    if not ((E == 0) | (E == 1)).all() and ((G == 0) | (G == 1)).all():
        raise ValueError('The matrices should be binary.')
    if G.dtype != bool:
        G = G.astype(bool)
    if E.dtype != bool:
        E = E.astype(bool)
    reversed = (E.T & G & (~E)).T
    FP = (E ^ reversed) & ~G
    m = len(G)
    F = m * (m - 1) * 0.5 - np.count_nonzero(G)
    return (FP.sum() + reversed.sum()) / max(F, 1)


def FDR(G: np.array, E: np.array) -> float:
    """Compute the False Discovery Rate between two graphs ((FP + reversed) / P)
    :param G: Ground-truth adjacency matrix
    :param E: Estimated adjacency Matrix
    :return: FDR between G and E
    """
    if not ((E == 0) | (E == 1)).all() and ((G == 0) | (G == 1)).all():
        raise ValueError('The matrices should be binary.')
    if G.dtype != bool:
        G = G.astype(bool)
    if E.dtype != bool:
        E = E.astype(bool)
    reversed = (E.T & G & (~E)).T
    FP = (E ^ reversed) & ~G
    return (FP.sum() + reversed.sum()) / max(E.sum(), 1)


def evaluate_graph(G: np.array, E: np.array, reverse_double=False) -> dict:
    """Compute the SHD, TPR, FPR and TPR.
    :param G: Ground-truth adjacency matrix
    :param E: Estimated adjacency Matrix
    :param reverse_double: a reversed edge should be counted as two false edges
    :return: FDR between G and E
    """
    return {
        'SHD': SHD(G, E, reverse_double),
        'TPR': TPR(G, E),
        'FPR': FPR(G, E),
        'FDR': FDR(G, E)
    }
