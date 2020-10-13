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

feature_filename = "/home/wangjue/biodata/DREAM5_network_inference_challenge/Network"+datasetname+"/input_data/expression_data.tsv"
# edge_filename    = "/home/wangjue/biodata/DREAM5_network_inference_challenge/Network"+datasetname+"/gold_standard/GoldStandard.tsv"

df = pd.read_csv(feature_filename,sep='\t')
df_t = df.T
df_t.to_csv('aracne'+datasetname+'.csv')

tfnum=334
if datasetname=='3':
    tfnum=334
elif datasetname=='4':
    tfnum=333

tflist=[]
for i in range(tfnum):
    tmpstr = 'G'+str(i+1)
    tflist.append(tmpstr)

with open('aracne_tf'+datasetname+'.txt','w') as fw:
    for item in tflist:
        fw.write("%s\n" % item)

