import sys
import os
import urllib
import gzip
import shutil


root = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(root, "tmp")

def create_bnlearn_dataset(name, n_samples, seed=None, save=False):
    name = name.lower()
    if not os.path.exists(CACHE):
        os.makedirs(CACHE)


