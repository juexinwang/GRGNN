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

# Preprocess network for sc
parser = argparse.ArgumentParser()
parser.add_argument('--network-name', type=str, default='ttrust',
                    help='ttrust')
parser.add_argument('--expression-name', type=str, default='TGFb', #1
                    help='TGFb from MAGIC or test')

args = parser.parse_args()

# Load gene expression into sparse matrix
def read_feature_file_sparse(filename, geneList):
    samplelist=[]
    featurelist=[]
    data =[]
    selectDict=[]
    selectList=[]
    count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            words = line.split()
            if count == -1:
                tcount =0
                for word in words:
                    selectDict[word] = tcount
                    tcount = tcount + 1
                for gene in geneList:
                    if gene in selectDict:
                        selectList.append(selectDict[gene])
                    else:
                        print(str(gene)+' is not in the input')
            if count >= 0:
                tmplist =[]
                for word in words:
                    tmplist.append(word)
                
                data_count = 0
                for item in selectList:
                    featurelist.append(count)
                    samplelist.append(data_count)
                    data.append(float(tmplist[item]))
                    data_count += 1
            count += 1
    f.close()
    feature = scipy.sparse.csr_matrix((data, (samplelist,featurelist)), shape=(len(samplelist),len(geneList)))   
    return feature

# Load gold standard edges into sparse matrix
# No edge types
# output mtx, geneList, geneDict, tfDict
def read_edge_file_csc(filename):
    row=[]
    col=[]
    data=[]
    # geneList, geneDict, tfDict
    geneList=[]
    geneDict={}
    tfDict={}

    count = 0
    genecount = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = words[0]
            end2 = words[1]

            #Add to sparse matrix
            row.append(end1)
            col.append(end2)
            data.append(1.0)
            row.append(end2)
            col.append(end1)
            data.append(1.0)

            # geneList, geneDict, tfDict
            if end1 not in geneDict:
                geneDict[end1] = genecount
                geneList.append(end1)
                genecount = genecount + 1
            if end2 not in geneDict:
                geneDict[end2] = genecount
                geneList.append(end2)
                genecount = genecount + 1
            if end1 not in tfDict:
                tfDict[end1] = ''
            count += 1
    f.close()
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(len(geneList), len(geneList)))
    return mtx, geneList, geneDict, tfDict

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
rowDict['1']=805
rowDict['3']=805
rowDict['4']=536
colDict['1']=1643
colDict['3']=4511
colDict['4']=5950


networkname=args.network_name
expressionname=arg.expression_name

if args.network_name=='ttrust':
    networkname = 'ttrust_rawdata.human.tsv'

if args.expression_name=='TGFb':
    expressionname = 'HMLE_TGFb_day_8_10.csv'
elif args.expression_name=='test':
    expressionname = 'test_data.csv'

edge_filename    = "data/"+networkname
feature_filename = "data/"+expressionname

graphcsc, geneList, geneDict, tfDict = read_edge_file_csc(edge_filename)
allx = read_feature_file_sparse(feature_filename, geneList)

pickle.dump(allx, open( "ind.dream"+expressionname+".allx", "wb" ) )
pickle.dump(graphcsc, open( "ind.dream"+expressionname+".csc", "wb" ) )

