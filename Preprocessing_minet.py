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

# Preprocess data for aracne.
# https://www.synapse.org/#!Synapse:syn3130840
parser = argparse.ArgumentParser()
parser.add_argument('--dream-num', type=str, default='3',
                    help='1 for In silico, 3 for E.coli, 4 for S. cerevisae')
args = parser.parse_args()

# Can be changed to dream3 and dream4
datasetname=args.dream_num

edge_filename    = "/home/wangjue/biodata/DREAM5_network_inference_challenge/Network"+datasetname+"/gold_standard/GoldStandard.tsv"

tfnum=334
samplenum=805
genenum=4511
if datasetname=='3':
    tfnum=334
    samplenum=805
    genenum=4511
elif datasetname=='4':
    tfnum=333
    samplenum=536
    genenum=5950

data = np.zeros((genenum,genenum))

with open(edge_filename) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        words = line.split()
        end1 = int(words[0].strip('G'))-1
        end2 = int(words[1].strip('G'))-1
        data[end1,end2]=1
        data[end2,end1]=1

np.savetxt('data/minet'+datasetname+'.csv', data, delimiter="\t",format='%10.5f')


# Running R codes