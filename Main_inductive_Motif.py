import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as cPickle
#import cPickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import math
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
sys.path.append('%s/software/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
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
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
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
attributes = geneexpression_attribute(allx,tfDict)
train_pos, train_neg, test_pos, test_neg = sample_neg_TF_motif(net, 1.0, max_train_num=args.max_train_num)

'''Train and apply classifier'''
A = net.copy()  # the observed network
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links

node_information = None
if args.use_embedding:
    embedding = generate_node2vec_embeddings(A, args.embedding_dim, True, train_neg) #?
    node_information = embedding

if args.use_attribute and attributes is not None: 
    if args.use_embedding:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

train_graphs, test_graphs, max_n_label = extractLinks2subgraphs_motif(A, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, node_information)
print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))


#DGCNN as the graph classifier
def DGCNN_classifer(train_graphs, test_graphs, train_node_information, max_n_label, set_epoch=50, eval_flag=True):
    # DGCNN configurations
    cmd_args.gm = 'DGCNN'
    cmd_args.sortpooling_k = 0.6
    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 2
    cmd_args.mode = 'gpu'
    cmd_args.num_epochs = set_epoch
    cmd_args.learning_rate = 1e-4
    cmd_args.batch_size = 50
    cmd_args.printAUC = True
    cmd_args.feat_dim = max_n_label + 1
    cmd_args.attr_dim = 0
    if train_node_information is not None:
        cmd_args.attr_dim = train_node_information.shape[1]
    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss, train_neg_idx, train_prob_results = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        test_loss=[]
        test_neg_idx=[]
        test_prob_results=[]
        if eval_flag:
            classifier.eval()
            test_loss, test_neg_idx, test_prob_results = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
            if not cmd_args.printAUC:
                test_loss[2] = 0.0
            print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))
    
    return test_loss, train_neg_idx, test_neg_idx, train_prob_results, test_prob_results

_, _, test_neg_agent0,  _,test_prob_agent0 =DGCNN_classifer(train_graphs, test_graphs, node_information, max_n_label, set_epoch = 10, eval_flag=True)


dic_agent0={}
for i in test_neg_agent0:
    dic_agent0[i]=0
tp0=0
fn0=0
tn0=0
fp0=0
testpos_size = len(test_pos[0])


testpos_size = len(test_pos[0])
for i in np.arange(len(test_prob_agent0)):
    if i<testpos_size: #positive part
        if i in dic_agent0 :
            fn0 = fn0 + 1
        else:
            tp0 = tp0 + 1
    else: #negative part
        if i in dic_agent0:
            fp0 = fp0 + 1
        else:
            tn0 = tn0 +1

print(str(tp0)+"\t"+str(fn0)+"\t"+str(tn0)+"\t"+str(fp0))

with open('acc_result.txt', 'a+') as f:
    f.write(allstr+"\t"+agent0_str+"\t"+agent1_str + '\n')




