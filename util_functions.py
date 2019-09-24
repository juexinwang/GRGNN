from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
#import cPickle as cp
import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/software/pytorch_DGCNN' % cur_dir)
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph
import node2vec

# return dictionary of transcription factors
def generateTFattribute(dream_name):
    tfDict={}
    number = 0
    if dream_name == 'dream1':
        number = 195
    elif dream_name == 'dream3':
        number = 334
    elif dream_name == 'dream4':
        number = 333
    for i in np.arange(number):
        tfDict[i]=1
    return tfDict    

# Use all the gene expression features
def geneexpression_attribute(allx, tfDict):
    tfAttr = np.zeros((len(allx),1))
    for i in np.arange(len(allx)) :
        if i in tfDict:
            tfAttr[i]=1.0
        else:
            tfAttr[i]=0.0
    trainAttributes = np.concatenate([allx, tfAttr], axis=1)
    return trainAttributes

# Generate explicit features for inductive learning, get trends features
def genenet_attribute(allx,tfNum):
    #1: average to one dimension
    allx_ = StandardScaler().fit_transform(allx)
    trainAttributes = np.average(allx_, axis=1).reshape((len(allx),1))
    #2: std,min,max as the attribute
    meanAtt = np.average(allx,axis=1).reshape((len(allx_),1))
    stdAtt = np.std(allx,axis=1).reshape((len(allx_),1))
    minVal = np.min(allx,axis=1).reshape((len(allx_),1))
    # expAtt = allx[:,:536]


    # #2 folder
    # qu1Val = np.quantile(allx,0.5, axis=1).reshape((len(allx_),1))
    # maxVal = np.max(allx,axis=1).reshape((len(allx_),1))

    # qu1Att = (qu1Val-minVal)/(maxVal-minVal)
    # qu2Att = (maxVal-qu1Val)/(maxVal-minVal)

    # quantilPerAtt =np.concatenate([qu1Att,qu2Att],axis=1)
    # quantilValAtt =np.concatenate([minVal, qu1Val, maxVal],axis=1)

    #4 folder
    qu1Val = np.quantile(allx,0.25, axis=1).reshape((len(allx_),1))
    qu2Val = np.quantile(allx,0.5, axis=1).reshape((len(allx_),1))
    qu3Val = np.quantile(allx,0.75, axis=1).reshape((len(allx_),1))
    maxVal = np.max(allx,axis=1).reshape((len(allx_),1))

    qu1Att = (qu1Val-minVal)/(maxVal-minVal)
    qu2Att = (qu2Val-qu1Val)/(maxVal-minVal)
    qu3Att = (qu3Val-qu2Val)/(maxVal-minVal)
    qu4Att = (maxVal-qu3Val)/(maxVal-minVal)

    quantilPerAtt =np.concatenate([qu1Att,qu2Att,qu3Att,qu4Att],axis=1)
    quantilValAtt =np.concatenate([minVal, qu1Val,qu2Val,qu3Val,maxVal],axis=1)


    #10 folder
    # qu1Val = np.quantile(allx,0.1, axis=1).reshape((len(allx_),1))
    # qu2Val = np.quantile(allx,0.2, axis=1).reshape((len(allx_),1))
    # qu3Val = np.quantile(allx,0.3, axis=1).reshape((len(allx_),1))
    # qu4Val = np.quantile(allx,0.4, axis=1).reshape((len(allx_),1))
    # qu5Val = np.quantile(allx,0.5, axis=1).reshape((len(allx_),1))
    # qu6Val = np.quantile(allx,0.6, axis=1).reshape((len(allx_),1))
    # qu7Val = np.quantile(allx,0.7, axis=1).reshape((len(allx_),1))
    # qu8Val = np.quantile(allx,0.8, axis=1).reshape((len(allx_),1))
    # qu9Val = np.quantile(allx,0.9, axis=1).reshape((len(allx_),1))
    # maxVal = np.max(allx,axis=1).reshape((len(allx_),1))

    # qu1Att = (qu1Val-minVal)/(maxVal-minVal)
    # qu2Att = (qu2Val-qu1Val)/(maxVal-minVal)
    # qu3Att = (qu3Val-qu2Val)/(maxVal-minVal)
    # qu4Att = (qu4Val-qu3Val)/(maxVal-minVal)
    # qu5Att = (qu5Val-qu4Val)/(maxVal-minVal)
    # qu6Att = (qu6Val-qu5Val)/(maxVal-minVal)
    # qu7Att = (qu7Val-qu6Val)/(maxVal-minVal)
    # qu8Att = (qu8Val-qu7Val)/(maxVal-minVal)
    # qu9Att = (qu9Val-qu8Val)/(maxVal-minVal)
    # qu10Att = (maxVal-qu9Val)/(maxVal-minVal)

    # quantilPerAtt =np.concatenate([qu1Att,qu2Att,qu3Att,qu4Att,qu5Att,qu6Att,qu7Att,qu8Att,qu9Att,qu10Att],axis=1)
    # quantilValAtt =np.concatenate([minVal, qu1Val,qu2Val,qu3Val,qu4Val, qu5Val, qu6Val, qu7Val, qu8Val, qu9Val, maxVal],axis=1)

    
    #5: TF or not, vital
    tfAttr = np.zeros((len(allx),1))
    for i in np.arange(tfNum) :
        tfAttr[i]=1.0
    
    #2. PCA to 3 dimensions
    allx_ = StandardScaler().fit_transform(allx)
    pca = PCA(n_components=3)
    pcaAttr = pca.fit_transform(allx_)

    # trainAttributes = np.concatenate([trainAttributes, stdAtt, minAtt, qu1Att, qu3Att, maxAtt, tfAttr], axis=1)
    #trainAttributes = np.concatenate([trainAttributes, stdAtt, tfAttr], axis=1)
    # Describe the slope
    # Best now:
    #trainAttributes = np.concatenate([trainAttributes, stdAtt, quantilPerAtt, tfAttr], axis=1)

    trainAttributes = np.concatenate([trainAttributes, stdAtt, quantilPerAtt, tfAttr], axis=1)
    # trainAttributes = np.concatenate([trainAttributes, stdAtt, quantilPerAtt, quantilValAtt, tfAttr], axis=1)
    
    #trainAttributes = np.concatenate([tfAttr], axis=1)
    
    return trainAttributes

# Negative sampling of the data, not restrict on TF
def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg


# Should only use this: only TF
def sample_neg_TF(net, test_ratio=0.1, TF_num=333, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    #TODO
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, TF_num), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

# Should only use this: only TF
def sample_neg_TF_motif(net, test_ratio=1.0, TF_num=333, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    #TODO
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, TF_num), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

# Should only use this: only TF in semi-supervise learning
def sample_neg_semi_TF(net, test_ratio=0.1, TF_num=333, train_pos=None, test_pos=None, max_train_num=None, semi_pool_fold=5):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    #TODO
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num * semi_pool_fold:
        i, j = random.randint(0, TF_num), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    train_neg  = (neg[0], neg[1])
    test_neg  = (neg[0], neg[1])
    return train_pos, train_neg, test_pos, test_neg

# Sample all possible links
def sample_neg_all(net):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    test_pos = (row, col)   
    # sample negative links for train/test
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for dataset')
    for i in np.arange(n):
        for j in np.arange(i+1,n):
            if net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
    test_neg = (neg[0], neg[1])
    return test_pos, test_neg

# Sample all possible links with TF
def sample_neg_all_TF(net,TF_num=333):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    test_pos = (row, col)   
    # sample negative links for train/test
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for dataset')
    for i in np.arange(TF_num):
        for j in np.arange(i+1,n):
            if net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
    test_neg = (neg[0], neg[1])
    return test_pos, test_neg

# Sample a very large number of negative links
def sample_neg_all_large(net,maximum_test=100000):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    test_pos = (row, col)   
    # sample negative links for train/test
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for dataset')
    recordDict={}
    while len(neg[0]) < maximum_test:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    test_neg = (neg[0], neg[1])
    return test_pos, test_neg

# Sample number of negative links in semi-supervise learning
def sample_neg_semi(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None, semi_pool_fold=5):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num * semi_pool_fold:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    #TODO: can be better later
    train_neg  = (neg[0], neg[1])
    test_neg  = (neg[0], neg[1])
    return train_pos, train_neg, test_pos, test_neg

# Extract subgraph from links for network motifs 
def extractLinks2subgraphs_motif(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto': # TODO
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label, node_information):
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
    print('Extract enclosed subgraph...')
    train_graphs = helper(A, train_pos, 1, node_information) + helper(A, train_neg, 0, node_information)
    test_graphs = helper(A, test_pos, 1, node_information) + helper(A, test_neg, 0, node_information)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']


# Extract subgraph from links 
def extractLinks2subgraphs(Atrain, Atest, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, train_node_information=None, test_node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto': # TODO
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label, node_information):
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
    print('Extract enclosed subgraph...')
    train_graphs = helper(Atrain, train_pos, 1, train_node_information) + helper(Atrain, train_neg, 0, train_node_information)
    test_graphs = helper(Atest, test_pos, 1, test_node_information) + helper(Atest, test_neg, 0, test_node_information)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']

# Extract subgraph from links for SVM
def extractLinks2subgraphsSVM(Atrain, Atest, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, train_node_information=None, test_node_information=None):
    # extract enclosing subgraphs
    def helper(A, links, g_label, node_information):
        g_list = []
        label_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            _, _, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            g_list.append(np.concatenate((n_features[0,:],n_features[1,:])))
            label_list.append(g_label)
        return g_list, label_list
    print('Extract enclosed subgraph...')
    train_graphs, train_labels = helper(Atrain, train_pos, 1, train_node_information) 
    train_graphs1, train_labels1 = helper(Atrain, train_neg, 0, train_node_information)
    train_graphs = train_graphs + train_graphs1
    train_labels = train_labels + train_labels1
    test_graphs, test_labels = helper(Atest, test_pos, 1, test_node_information)
    test_graphs1, test_labels1 = helper(Atest, test_neg, 0, test_node_information)
    test_graphs = test_graphs + test_graphs1
    test_labels = test_labels + test_labels1
    return train_graphs, test_graphs, train_labels, test_labels

# Add labels
def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, labels.tolist(), features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

    
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, 
            workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

