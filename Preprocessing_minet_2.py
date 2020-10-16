import time
import numpy as np
import argparse
from copy import deepcopy
from scipy import interpolate
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import scipy.sparse
import sys
import pickle
import pandas as pd

# Preprocess data for minet.
# https://www.bioconductor.org/packages/release/bioc/html/minet.html
# https://www.synapse.org/#!Synapse:syn3130840
parser = argparse.ArgumentParser()
parser.add_argument('--dream-num', type=str, default='3',
                    help='1 for In silico, 3 for E.coli, 4 for S. cerevisae')
args = parser.parse_args()

# Can be changed to dream3 and dream4
datasetname=args.dream_num

# save and load
def readnpy(input):
    tlist=[]
    with open(input) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = float(line)
            tlist.append(line)
        f.close()
    return tlist

# mrnet
mrlist=readnpy("data/mr"+datasetname+".csv")
mr_true=readnpy("data/mr_true"+datasetname+".csv")

np.save("data/mr"+datasetname+".npy",mrlist)
np.save("data/mr_true"+datasetname+".npy",mr_true)

# aracne
mrlist=readnpy("data/ar"+datasetname+".csv")
mr_true=readnpy("data/ar_true"+datasetname+".csv")

np.save("data/ar"+datasetname+".npy",mrlist)
np.save("data/ar_true"+datasetname+".npy",mr_true)

# clr
mrlist=readnpy("data/clr"+datasetname+".csv")
mr_true=readnpy("data/clr_true"+datasetname+".csv")

np.save("data/clr"+datasetname+".npy",mrlist)
np.save("data/clr_true"+datasetname+".npy",mr_true)
