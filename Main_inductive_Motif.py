import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import math
import argparse
from util_functions import *

# Test network motifs
# Only one agent is used
parser = argparse.ArgumentParser(description='Gene Regulatory Graph Neural Network in network motifs')
# general settings
parser.add_argument('--traindata-name', default='dream3', help='train network name')
parser.add_argument('--testdata-name', default=None, help='test network name, usually transductive here')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--training-ratio', type=float, default=1.0,
                    help='ratio of used training set')
# Dimension of embedding
parser.add_argument('--embedding-dim', type=int, default=1,
                    help='embedding dimmension')
# model settings
parser.add_argument('--hop', type=int, default=1,  
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=True,
                    help='whether to use node attributes')
args = parser.parse_args()

print(args)

if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

# dream1: top 195
# dream3: top 334
# dream4: top 333 are TF
dreamTFdict={}
dreamTFdict['dream1']=195
dreamTFdict['dream3']=334
dreamTFdict['dream4']=333

tfDict=generateTFattribute(args.traindata_name)

#Transductive learning
# mainly used here on the motif
net = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.traindata_name)))
group = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.traindata_name)))
allx =group.toarray().astype('float32')
#deal with the features:
# attributes = geneexpression_attribute(allx,tfDict)
#deal with the features, zscore and features:
attributes = geneexpression_attribute_zscore(allx,tfDict)
train_pos, train_neg = sample_neg_TF_motif(net, TF_num=dreamTFdict[args.traindata_name])

'''Train and apply classifier'''
A = net.copy()  # the observed network
# A[test_pos[0], test_pos[1]] = 0  # mask test links
# A[test_pos[1], test_pos[0]] = 0  # mask test links

node_information = None
if args.use_embedding:
    embedding = generate_node2vec_embeddings(A, args.embedding_dim, True, train_neg) #?
    node_information = embedding

if args.use_attribute and attributes is not None: 
    if args.use_embedding:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

pos_graphs_labels,pos_graphs_features,neg_graphs_labels,neg_graphs_features, max_n_label = extractLinks2subgraphs_motif(A, train_pos, train_neg, args.hop, args.max_nodes_per_hop, node_information, dreamTFdict[args.traindata_name])
print('# pos: %d, # neg: %d' % (len(pos_graphs_labels), len(neg_graphs_labels)))

# np.save('pos_graphs_labels.npy',pos_graphs_labels)
# np.save('pos_graphs_features.npy',pos_graphs_features)
np.save('neg_graphs_labels.npy',neg_graphs_labels)
np.save('neg_graphs_features_zscore.npy',neg_graphs_features)
# pickle.dump( pos_graphs, open( "pos_graphs.pickle", "wb" ) )
# pickle.dump( neg_graphs, open( "neg_graphs.pickle", "wb" ) )