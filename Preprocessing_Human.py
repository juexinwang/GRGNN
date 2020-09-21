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

# Preprocess Human data from Official DoRothEA websites.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Human',
                    help='Human or Mouse')
parser.add_argument('--pearson_net', type=float, default=0.8, #1
                    help='pearson correlation as the network')
parser.add_argument('--mutual_net', type=int, default=3, #3
                    help='mutual information as the network')
parser.add_argument('--random_net', type=float, default=0.003, #3
                    help='random as the network')

args = parser.parse_args()

# get Dict with key as gene name
def generateGeneDict(feature_filename):
    linecount = -1
    geneDict = {}
    with open(feature_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            if linecount == -1:
                count = 0
                for word in words:
                    geneDict[word] = count
                    count +=1
            linecount+=1
        f.close()
    return geneDict

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
def read_edge_file_csc(filename, geneDict, sample_size):
    row=[]
    col=[]
    data=[]
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            if words[0] in geneDict and words[1] in geneDict:
                end1 = int(geneDict[words[0]])
                end2 = int(geneDict[words[1]])
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

# calculate random
def randomMatrix(data, threshold=0.003):
    row=[]
    col=[]
    edata=[]
    for i in np.arange(data.shape[1]):
        for j in np.arange(data.shape[1]):
            if np.random.random_sample() <= threshold:
                row.append(i)
                col.append(j)
                edata.append(1.0)
    
    row = np.asarray(row)
    col = np.asarray(col)
    edata = np.asarray(edata)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((edata, (row, col)), shape=(data.shape[1], data.shape[1]))
    return mtx

# Setting of Dream dataset size
rowDict={}
colDict={}
rowDict['Human']=1131
colDict['Human']=15276
#human TF: 747

# Can be changed to dream3 and dream4
datasetname=args.dataset

feature_filename = "data/Human_expression.txt"
edge_filename    = "data/Human_goldstandard.txt"

geneDict = generateGeneDict(feature_filename)
graphcsc = read_edge_file_csc(edge_filename, geneDict, sample_size=15276)
graphcsc = read_edge_file_csc(edge_filename, geneDict, sample_size=15276)
allx = read_feature_file_sparse(feature_filename, sample_size=15276, feature_size=1131)

pickle.dump(allx, open( "data/ind."+datasetname+".allx", "wb" ) )
pickle.dump(graphcsc, open( "data/ind."+datasetname+".csc", "wb" ) )

# cross validation
edge_filename1    = "data/Human_goldstandard1.txt"
edge_filename2    = "data/Human_goldstandard2.txt"
edge_filename3    = "data/Human_goldstandard3.txt"
edge_filename23   = "data/Human_goldstandard23.txt"
edge_filename13   = "data/Human_goldstandard13.txt"
edge_filename12   = "data/Human_goldstandard12.txt"
graphcsc1 = read_edge_file_csc(edge_filename1, geneDict, sample_size=15276)
graphcsc2 = read_edge_file_csc(edge_filename2, geneDict, sample_size=15276)
graphcsc3 = read_edge_file_csc(edge_filename3, geneDict, sample_size=15276)
graphcsc23 = read_edge_file_csc(edge_filename23, geneDict, sample_size=15276)
graphcsc13 = read_edge_file_csc(edge_filename13, geneDict, sample_size=15276)
graphcsc12 = read_edge_file_csc(edge_filename12, geneDict, sample_size=15276)

pickle.dump(graphcsc1, open( "data/ind."+datasetname+"1.csc", "wb" ) )
pickle.dump(graphcsc2, open( "data/ind."+datasetname+"2.csc", "wb" ) )
pickle.dump(graphcsc3, open( "data/ind."+datasetname+"3.csc", "wb" ) )
pickle.dump(graphcsc23, open( "data/ind."+datasetname+"23.csc", "wb" ) )
pickle.dump(graphcsc13, open( "data/ind."+datasetname+"13.csc", "wb" ) )
pickle.dump(graphcsc12, open( "data/ind."+datasetname+"12.csc", "wb" ) )

# # data as the correlation
# rownum = rowDict[datasetname]    
# colnum = colDict[datasetname]
# data = np.zeros((rownum,colnum))

# count = -1
# with open(feature_filename) as f:
#     lines = f.readlines()
#     for line in lines:
#         if count >= 0:
#             line = line.strip()
#             words = line.split()
#             ncount = 0
#             for word in words:
#                 data[count, ncount] = word
#                 ncount = ncount + 1
#         count = count + 1
#     f.close()

# # Calculate Pearson's Correlation coeficient
# pmatrix = pearsonMatrix(data, args.pearson_net)
# np.save('data/'+datasetname+'_pmatrix_'+str(args.pearson_net)+'.npy', pmatrix)

# # Calculate Mutual Information
# mmatrix = mutualMatrix(data, args.mutual_net)
# np.save('data/'+datasetname+'_mmatrix_'+str(args.mutual_net)+'.npy', mmatrix)

# # Calculate a random network based on 
# rmatrix = randomMatrix(data, args.random_net)
# np.save('data/'+datasetname+'_rmatrix_'+str(args.random_net)+'.npy', rmatrix)