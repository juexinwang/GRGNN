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
parser.add_argument('--dream-num', type=str, default='3',
                    help='1 for In silico, 3 for E.coli, 4 for S. cerevisae')
parser.add_argument('--pearson_net', type=float, default=0.8, #1
                    help='pearson correlation as the network')
parser.add_argument('--mutual_net', type=int, default=3, #3
                    help='mutual information as the network')
parser.add_argument('--random_net', type=float, default=0.003, #3
                    help='random as the network')

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

# cross validation Load gold standard edges into sparse matrix
def read_edge_file_csc_CV(filename, sample_size):
    #cv=3
    count = 0
    tfDict ={}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            tfDict[words[1]]=''
            count += 1
    f.close()
    tfcount = len(tfDict)
    print('Total tf number is '+str(tfcount))
    tfNum = int(tfcount/3)

    row1=[]
    row2=[]
    row3=[]
    col1=[]
    col2=[]
    col3=[]
    data1=[]
    data2=[]
    data3=[]
    count = 0
    tfDictTmp = {}
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
            if words[1] in tfDict:
                tfDictTmp[words[1]]=''
            if int(len(tfDictTmp)/tfNum) == 0:
                row1.append(end1)
                col1.append(end2)
                data1.append(1.0)
                row1.append(end2)
                col1.append(end1)
                data1.append(1.0)
            elif int(len(tfDictTmp)/tfNum) == 1:
                row2.append(end1)
                col2.append(end2)
                data2.append(1.0)
                row2.append(end2)
                col2.append(end1)
                data2.append(1.0)
            else:
                row3.append(end1)
                col3.append(end2)
                data3.append(1.0)
                row3.append(end2)
                col3.append(end1)
                data3.append(1.0)
            count += 1
    f.close()

    row1_ = np.asarray(row1)
    row2_ = np.asarray(row2)
    row3_ = np.asarray(row3)
    col1_ = np.asarray(col1)
    col2_ = np.asarray(col2)
    col3_ = np.asarray(col3)
    data1_ = np.asarray(data1)
    data2_ = np.asarray(data2)
    data3_ = np.asarray(data3)

    row23_ = np.asarray(row2+row3)
    row13_ = np.asarray(row1+row3)
    row12_ = np.asarray(row1+row2)
    col23_ = np.asarray(col2+col3)
    col13_ = np.asarray(col1+col3)
    col12_ = np.asarray(col1+col2)
    data23_ = np.asarray(data2+data3)
    data13_ = np.asarray(data1+data3)
    data12_ = np.asarray(data1+data2)
    #check and get full matrix
    mtx1 = scipy.sparse.csc_matrix((data1_, (row1_, col1_)), shape=(sample_size, sample_size))
    mtx2 = scipy.sparse.csc_matrix((data2_, (row2_, col2_)), shape=(sample_size, sample_size))
    mtx3 = scipy.sparse.csc_matrix((data3_, (row3_, col3_)), shape=(sample_size, sample_size))
    mtx23 = scipy.sparse.csc_matrix((data23_, (row23_, col23_)), shape=(sample_size, sample_size))
    mtx13 = scipy.sparse.csc_matrix((data13_, (row13_, col13_)), shape=(sample_size, sample_size))
    mtx12 = scipy.sparse.csc_matrix((data12_, (row12_, col12_)), shape=(sample_size, sample_size))
    return mtx1,mtx2,mtx3,mtx23,mtx13,mtx12

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

# Can be changed to dream3 and dream4
datasetname=args.dream_num

feature_filename = "/home/wangjue/biodata/DREAM5_network_inference_challenge/Network"+datasetname+"/input_data/expression_data.tsv"
edge_filename    = "/home/wangjue/biodata/DREAM5_network_inference_challenge/Network"+datasetname+"/gold_standard/GoldStandard.tsv"

# graphcsc = read_edge_file_csc(edge_filename, sample_size=args.sample_size)
# allx = read_feature_file_sparse(feature_filename, sample_size=args.sample_size, feature_size=args.feature_size)

# pickle.dump(allx, open( "ind.dream"+datasetname+".allx", "wb" ) )
# pickle.dump(graphcsc, open( "ind.dream"+datasetname+".csc", "wb" ) )

sample_size=4511
if args.dream_num == '3':
    sample_size=4511
elif args.dream_num == '4':
    sample_size=5950

graphcsc1,graphcsc2,graphcsc3,graphcsc23,graphcsc13,graphcsc12 = read_edge_file_csc_CV(edge_filename, sample_size)
pickle.dump(graphcsc1, open( "data/dream/ind.data"+datasetname+"_1.csc", "wb" ) )
pickle.dump(graphcsc2, open( "data/dream/ind.data"+datasetname+"_2.csc", "wb" ) )
pickle.dump(graphcsc3, open( "data/dream/ind.data"+datasetname+"_3.csc", "wb" ) )
pickle.dump(graphcsc23, open( "data/dream/ind.data"+datasetname+"_23.csc", "wb" ) )
pickle.dump(graphcsc13, open( "data/dream/ind.data"+datasetname+"_13.csc", "wb" ) )
pickle.dump(graphcsc12, open( "data/dream/ind.data"+datasetname+"_12.csc", "wb" ) )

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
# np.save('data'+datasetname+'_pmatrix_'+str(args.pearson_net)+'.npy', pmatrix)

# # Calculate Mutual Information
# mmatrix = mutualMatrix(data, args.mutual_net)
# np.save('data'+datasetname+'_mmatrix_'+str(args.mutual_net)+'.npy', mmatrix)

# # Calculate a random network based on 
# rmatrix = randomMatrix(data, args.random_net)
# np.save('data'+datasetname+'_rmatrix_'+str(args.random_net)+'.npy', rmatrix)