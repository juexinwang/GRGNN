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

# Preprocess DREAM5 data from Official DREAM websites.
# https://www.synapse.org/#!Synapse:syn3130840
parser = argparse.ArgumentParser()
parser.add_argument('--dream-num', type=str, default='1',
                    help='1 for In silico, 3 for E.coli, 4 for S. cerevisae')
parser.add_argument('--pearson_net', type=float, default=0.8, #1
                    help='pearson correlation as the network')
parser.add_argument('--mutual_net', type=int, default=3, #3
                    help='mutual information as the network')

args = parser.parse_args()

# Load gene expression into sparse matrix
def read_feature_file_sparse(filename, sample_size, feature_size):
    samplelist=[]
    featurelist=[]
    data =[]
    count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if count >= 0:
                line = line.strip()
                words = line.split()
                data_count = 0
                for word in words:
                    featurelist.append(count)
                    samplelist.append(data_count)
                    data.append(float(word))
                    data_count += 1
            count += 1
    f.close()
    feature = scipy.sparse.csr_matrix((data, (samplelist,featurelist)), shape=(sample_size,feature_size))   
    return feature

# Load gold standard edges into sparse matrix
def read_edge_file_csc(filename, sample_size):
    row=[]
    col=[]
    data=[]
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = int(words[0][1:])-1
            end2 = int(words[1][1:])-1
            if end1 > end2:
                tmpp = end1
                end1 = end2
                end2 = tmpp
            row.append(end1)
            col.append(end2)
            data.append(1.0)
            row.append(end2)
            col.append(end1)
            data.append(1.0)
            count += 1
    f.close()
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(sample_size, sample_size))
    return mtx

# calculate Pearson's Correlation coefficient of gene expression
def pearsonMatrix(data, threshold=0.8):
    row=[]
    col=[]
    edata=[]
    for i in np.arange(data.shape[1]):
        for j in np.arange(data.shape[1]):
            corr, _ = pearsonr(data[:,i], data[:,j])
            if abs(corr) >= threshold:
                row.append(i)
                col.append(j)
                edata.append(1.0)
    
    row = np.asarray(row)
    col = np.asarray(col)
    edata = np.asarray(edata)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((edata, (row, col)), shape=(data.shape[1], data.shape[1]))
    return mtx

# calculate Mutual Information of gene expression 
def mutualMatrix(data, thresholdfold=3, bin=100):
    row=[]
    col=[]
    edata=[]
    total = []
    for i in np.arange(data.shape[1]):
        for j in np.arange(data.shape[1]):
            total.append(calc_MI(data[:,i], data[:,j], bin))
    
    total = np.asarray(total)
    threshold = np.mean(total)+thresholdfold*np.std(total)

    for i in np.arange(data.shape[1]):
        for j in np.arange(data.shape[1]):
            if calc_MI(data[:,i], data[:,j], bin) >= threshold:
                row.append(i)
                col.append(j)
                edata.append(1.0)
    
    row = np.asarray(row)
    col = np.asarray(col)
    edata = np.asarray(edata)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((edata, (row, col)), shape=(data.shape[1], data.shape[1]))
    return mtx

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# Setting of Dream dataset size
rowDict={}
colDict={}
rowDict['1']=805
rowDict['3']=805
rowDict['4']=536
colDict['1']=1643
colDict['3']=4511
colDict['4']=5950

# Can be changed to dream3 and dream4
datasetname=args.dream_num

feature_filename = "~/biodata/DREAM5_network_inference_challenge/Network"+datasetname+"/input_data/expression_data.tsv"
edge_filename    = "~/biodata/DREAM5_network_inference_challenge/Network"+datasetname+"/gold_standard/gold_standard_signed.tsv"

graphcsc = read_edge_file_csc(edge_filename, sample_size=args.sample_size)
allx = read_feature_file_sparse(feature_filename, sample_size=args.sample_size, feature_size=args.feature_size)

pickle.dump(allx, open( "ind.dream"+datasetname+".allx", "wb" ) )
pickle.dump(graphcsc, open( "ind.dream"+datasetname+".csc", "wb" ) )

# data as the correlation
rownum = rowDict[datasetname]    
colnum = colDict[datasetname]
data = np.zeros((rownum,colnum))

count = -1
with open(feature_filename) as f:
    lines = f.readlines()
    for line in lines:
        if count >= 0:
            line = line.strip()
            words = line.split()
            ncount = 0
            for word in words:
                data[count, ncount] = word
                ncount = ncount + 1
        count = count + 1
    f.close()

# Calculate Pearson's Correlation coeficient
pmatrix = pearsonMatrix(data, args.pearson_net)
np.save('dream'+datasetname+'_pmatrix_'+args.pearson_net+'.npy', pmatrix)

# Calculate Mutual Information
mmatrix = mutualMatrix(data, args.mutual_net)
np.save('dream'+datasetname+'_mmatrix_'+args.mutual_net+'.npy', mmatrix)
