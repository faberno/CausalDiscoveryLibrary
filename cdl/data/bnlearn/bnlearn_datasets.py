import glob
import os
import urllib.request
import json
import random

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from .R_functions import load, set_seed, amat, rbn, as_dataframe

BN_REPO = [
    "asia", "cancer", "earthquake", "sachs", "survey",  # small
    "alarm", "barley", "child", "insurance", "mildew", "water"  # medium
    "hailfinder", "hepar2", "win95pts",  # large
    "andes", "diabetes", "link", "munin", "pathfinder", "pigs"  # very large
]

BN_URL = "https://www.bnlearn.com/bnrepository/"

root = os.path.dirname(os.path.abspath(__file__))
resources = os.path.join(root, "resources")


def find_dataset_index(dir):
    i = 0
    while len(glob.glob(os.path.join(dir, f"{i}_*"))) != 0:
        i += 1
    return i

def create_bnlearn_dataset(name, n_samples, seed=None, save=False):
    name = name.lower()
    assert(name in BN_REPO)
    if not os.path.exists(resources):
        os.makedirs(resources)

    repo_path = os.path.join(resources, name)
    file_path = os.path.join(repo_path, f'{name}.rda')
    if not os.path.exists(repo_path):
        os.makedirs(repo_path)
    if not os.path.isfile(file_path):
        urllib.request.urlretrieve(f"{BN_URL}{name}/{name}.rda", file_path)

    if seed is None:
        seed = random.randint(0, 10000)
    set_seed(seed)
    load(file_path)
    bn = robjects.globalenv.find("bn")
    adj = as_dataframe(amat(bn))
    X = rbn(bn, n_samples)
    variable_states = {name: list(entry[3].names[0]) for (name, entry) in zip(bn.names, bn)}
    with robjects.default_converter + pandas2ri.converter:
        adj = robjects.conversion.get_conversion().rpy2py(adj)
        X = robjects.conversion.get_conversion().rpy2py(X)

    graph_path = os.path.join(repo_path, f"{name}_graph.csv")
    if not os.path.isfile(graph_path):
        adj.to_csv(graph_path)

    state_path = os.path.join(repo_path, f"{name}_states.json")
    if not os.path.isfile(state_path):
        with open(state_path, 'w') as fp:
            json.dump(variable_states, fp)

    if save:
        dataset_path = os.path.join(repo_path, "syn_datasets")
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            i = 0
        else:
            i = find_dataset_index(dataset_path)
        X.to_csv(os.path.join(dataset_path, f"{i}_{name}_{n_samples}_{seed}.csv"), index=False)

    return adj, X, variable_states

def load_bnlearn_dataset(name, index):
    name = name.lower()
    repo_path = os.path.join(resources, name)
    graph_path = os.path.join(repo_path, f"{name}_graph.csv")
    state_path = os.path.join(repo_path, f"{name}_states.json")
    dataset_path = os.path.join(repo_path, "syn_datasets", f"{index}_{name}_*.csv")
    candidates = glob.glob(dataset_path)
    assert(len(candidates) > 0, f"No dataset with index {index} was found.")
    dataset_path = candidates[0]
    assert(
        all((os.path.isfile(graph_path),
             os.path.isfile(state_path),
             os.path.isfile(dataset_path)))
    )
    adj = pd.read_csv(graph_path, index_col=0)
    X = pd.read_csv(dataset_path)
    with open(state_path, 'r') as fp:
        states = json.load(fp)

    return adj, X, states



def main():
    # create_bnlearn_dataset("asia", seed=2, save=True)
    a = load_bnlearn_dataset("asia", 0)
    print("")

if __name__ == "__main__":
    main()