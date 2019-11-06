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
import csv

# Preprocess network for sc
parser = argparse.ArgumentParser()
parser.add_argument('--network-name', type=str, default='ttrust',
                    help='ttrust')
parser.add_argument('--expression-name', type=str, default='TGFb', #1
                    help='TGFb from MAGIC or test')

args = parser.parse_args()

# Read network and expression
# output geneList, geneDict
def preprocess_network(edge_filename, feature_filename):

    # geneList, geneDict
    geneList=[]
    tList=[]
    tgeneDict={}
    geneDict={}

    genecount = 0
    with open(edge_filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = words[0]
            end2 = words[1]

            # geneList, geneDict, tfDict
            if end1 not in tgeneDict:
                tgeneDict[end1] = genecount
                tList.append(end1)
                genecount = genecount + 1
            if end2 not in tgeneDict:
                tgeneDict[end2] = genecount
                tList.append(end2)
                genecount = genecount + 1
            # if end1 not in tfDict:
            #     tfDict[end1] = tgeneDict[end1]
    f.close()

    count = 0
    exDict={}
    with open(feature_filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            words = line.split(',')
            if count == 0:
                tcount =0
                for word in words:
                    exDict[word] = tcount
                    tcount = tcount + 1
            else:
                break
            count = count+1
    f.close()

    count = 0
    for gene in tList:
        if gene in exDict:
            geneList.append(gene)
            geneDict[gene] = count
            count +=1

    return geneList, geneDict

# Load gold standard edges into sparse matrix
# No edge types
# output mtx, tfDict
# Additional outfile for matlab
def read_edge_file_csc(filename, geneDict):
    row=[]
    col=[]
    data=[]

    tfDict={}
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = words[0]
            end2 = words[1]
            
            #Add to sparse matrix
            if end1 in geneDict and end2 in geneDict:
                row.append(geneDict[end1])
                col.append(geneDict[end2])
                data.append(1.0)
                row.append(geneDict[end2])
                col.append(geneDict[end1])
                data.append(1.0)
                count += 1
                if end1 not in tfDict:
                    tfDict[end1] = geneDict[end1]
    f.close()
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(len(geneDict), len(geneDict)))
    
    #python output
    # return mtx, tfDict

    #Output for matlab
    return mtx, tfDict, row, col, data

# Load gene expression into sparse matrix
def read_feature_file_sparse(filename, geneList, geneDict):
    samplelist=[]
    featurelist=[]
    data =[]
    selectDict={}
    selectList=[]
    count = -1

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            words = line.split(',')
            if count == -1:
                tcount =0
                for word in words:
                    if word in geneDict:
                        selectDict[word] = tcount
                    tcount = tcount + 1
                ntcount = 0
                ytcount = 0
                for gene in geneList:
                    if gene in selectDict:
                        selectList.append(selectDict[gene])
                        ytcount += 1
                    else:
                        print(str(gene)+' is not in the input')
                        ntcount += 1
                print(str(ytcount)+"\t"+str(ntcount))
            if count >= 0:
                tmplist =[]
                for word in words:
                    tmplist.append(float(word))
                avgtmp = np.sum(tmplist)/float(len(tmplist))
                
                data_count = 0
                for item in selectList:
                    samplelist.append(count)
                    featurelist.append(data_count)
                    # data.append(float(tmplist[item]))
                    if tmplist[item]>=avgtmp:
                        data.append(1)
                    else:
                        data.append(0)
                    data_count += 1
            count += 1
    f.close()
    # As dream: rows as genes, columns as samples: This is transpose of the original scRNA data
    feature = scipy.sparse.csr_matrix((data, (featurelist, samplelist)), shape=(len(selectList),count))  

    # For Matlab
    dim2out = [[0.0] * len(selectList) for i in range(count)]
    count = -1
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:            
            line = line.strip()
            words = line.split(',')
            if count >= 0:
                tmplist =[]
                for word in words:
                    tmplist.append(float(word))
                avgtmp = np.sum(tmplist)/float(len(tmplist))
                
                data_count = 0
                for item in selectList:
                    dim2out[count][data_count]=float(tmplist[item])
                    data_count += 1
            count += 1
    f.close()

    return feature, dim2out



def read_edge_file_dict(filename, geneDict):
    graphdict={}
    tdict={}
    count = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            end1 = words[0]
            end2 = words[1]
            if end1 in geneDict and end2 in geneDict:
                tdict[geneDict[end1]]=""
                tdict[geneDict[end2]]=""
                if geneDict[end1] in graphdict:
                    tmplist = graphdict[geneDict[end1]]
                else:
                    tmplist = []
                tmplist.append(geneDict[end2])
                graphdict[geneDict[end1]]= tmplist
            count += 1
    f.close()
    #check and get full matrix
    for i in range(len(geneDict)):
        if i not in tdict:
            graphdict[i]=[]
    return graphdict

networkname=args.network_name
expressionname=args.expression_name

if args.network_name=='ttrust':
    networkname = 'trrust_rawdata.human.tsv'

if args.expression_name=='TGFb':
    expressionname = 'HMLE_TGFb_day_8_10.csv'
    # expressionname = 'HMLE_TGFb_day_8_10_part.csv'
elif args.expression_name=='test':
    expressionname = 'test_data.csv'

edge_filename    = "/home/wangjue/biodata/scData/network/"+networkname
feature_filename = "/home/wangjue/biodata/scData/"+expressionname

geneList, geneDict = preprocess_network(edge_filename, feature_filename)

#only python
# graphcsc, tfDict = read_edge_file_csc(edge_filename, geneDict)
# feature = read_feature_file_sparse(feature_filename, geneList, geneDict)

#python and matlab
graphcsc, tfDict, rowO, colO, dataO  = read_edge_file_csc(edge_filename, geneDict)
feature, dim2out = read_feature_file_sparse(feature_filename, geneList, geneDict)

graphdict = read_edge_file_dict(edge_filename, geneDict)

x = feature[0:100]
tx = feature[0:100]
allx = feature[100:]
# allx = feature
testindex = ""
for i in range(100):
    testindex = testindex + str(i) + "\n"


with open("data/sc/gene.txt",'w') as fw:
    count = 0
    for item in geneList:
        fw.write(str(count)+"\t"+item+"\n")
        count += 1

with open("data/sc/TF.txt",'w') as fw:
    count = 0
    for key in tfDict:
        fw.write(key+"\t"+str(tfDict[key])+"\n")
        count += 1

pickle.dump(allx, open( "data/sc/ind."+args.expression_name+".allx", "wb" ) )
pickle.dump(graphcsc, open( "data/sc/ind."+args.expression_name+".csc", "wb" ) )

pickle.dump(x, open( "data/sc/ind."+args.expression_name+".x", "wb" ) )
pickle.dump(tx, open( "data/sc/ind."+args.expression_name+".tx", "wb" ) )
pickle.dump(graphdict, open( "data/sc/ind."+args.expression_name+".graph", "wb" ) )
with open ("data/sc/ind."+args.expression_name+".test.index", 'w') as fw:
    fw.writelines(testindex)
    fw.close()


# For matlab
with open('data/sc/'+args.expression_name+'.features.csv','w') as fw:
    writer = csv.writer(fw)
    writer.writerows(dim2out)
fw.close()

with open('data/sc/'+args.expression_name+'.row.csv','w') as fw:
    for item in rowO:
        fw.write(str(item)+"\n")
fw.close()

with open('data/sc/'+args.expression_name+'.col.csv','w') as fw:
    for item in colO:
        fw.write(str(item)+"\n")
fw.close()

with open('data/sc/'+args.expression_name+'.data.csv','w') as fw:
    for item in dataO:
        fw.write(str(item)+"\n")
fw.close()

