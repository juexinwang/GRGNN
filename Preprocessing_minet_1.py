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
# python Preprocessing_minet_1.py --dream-num 4
# Rscript Prepocessing_minet.r 3
# python Preprocessing_minet_2.py --dream-num 4
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

#directly save 
# np.savetxt('data/minet'+datasetname+'.csv', data, delimiter="\t",fmt="%10.5f")

genelist=[]
for i in range(genenum):
    tmpstr = 'G'+str(i+1)
    genelist.append(tmpstr)

# All
df = pd.DataFrame(data,index=genelist,columns=genelist)
df.to_csv('data/True'+datasetname+'.csv',sep='\t')

# Running R codes
# Rscript Preprocessing_minet.r 3
