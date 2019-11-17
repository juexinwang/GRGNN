import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as cPickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
sys.path.append('%s/software/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *


parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# Data from http://dreamchallenges.org/project/dream-5-network-inference-challenge/
# data1: In silico
# data3: E. coli
# data4: Yeast
# general settings
parser.add_argument('--data-name', default='USAir', help='network name')
parser.add_argument('--traindata-name', default='data3', help='train network name')
parser.add_argument('--traindata-name2', default=None, help='train network name2')
parser.add_argument('--testdata-name', default='data4', help='test network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
# Pearson correlation
parser.add_argument('--embedding-dim', type=int, default=1,
                    help='embedding dimmension')
parser.add_argument('--pearson_net', type=float, default=0.8,
                    help='pearson correlation as the network')
# model settings
parser.add_argument('--hop', default=0, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=True,
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
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

# data1: top 195 are TF
# data3: top 334 are TF
# data4: top 333 are TF
dreamTFdict={}
dreamTFdict['data1']=195
dreamTFdict['data3']=334
dreamTFdict['data4']=333

# Inductive learning
# Training on 1 data, test on 1 data
if args.traindata_name is not None:
    trainNet_ori = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.traindata_name)))
    trainGroup = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.traindata_name)))
    trainNet = np.load(args.file_dir+'/data/dream/'+args.traindata_name+'_pmatrix_'+str(args.pearson_net)+'.npy').tolist()
    # trainNet = np.load(args.file_dir+'/data/dream/'+args.traindata_name+'_mmatrix_'+str(args.pearson_net)+'.npy').tolist()
    allx =trainGroup.toarray().astype('float32')
    #deal with the features:
    trainAttributes = genenet_attribute(allx,dreamTFdict[args.traindata_name])   

    testNet_ori = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.testdata_name)))
    testGroup = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.testdata_name)))
    testNet = np.load(args.file_dir+'/data/dream/'+args.testdata_name+'_pmatrix_'+str(args.pearson_net)+'.npy').tolist()
    # testNet = np.load(args.file_dir+'/data/dream/'+args.testdata_name+'_mmatrix_'+str(args.pearson_net)+'.npy').tolist()
    allxt =testGroup.toarray().astype('float32')
    #deal with the features:
    testAttributes = genenet_attribute(allxt,dreamTFdict[args.testdata_name])

    # train_pos, train_neg, _, _ = sample_neg(trainNet_ori, 0.0, max_train_num=args.max_train_num)
    train_pos, train_neg, _, _ = sample_neg_TF(trainNet_ori, 0.0, TF_num=dreamTFdict[args.traindata_name], max_train_num=args.max_train_num)
    
    #_, _, test_pos, test_neg = sample_neg(testNet_ori, 1.0, max_train_num=args.max_train_num)
    _, _, test_pos, test_neg = sample_neg_TF(testNet_ori, 1.0, TF_num=dreamTFdict[args.testdata_name], max_train_num=args.max_train_num)
    # test_pos, test_neg = sample_neg_all_TF(testNet_ori, TF_num=dreamTFdict[args.testdata_name])



'''Train and apply classifier'''
Atrain = trainNet.copy()  # the observed network
Atest = testNet.copy()  # the observed network
Atest[test_pos[0], test_pos[1]] = 0  # mask test links
Atest[test_pos[1], test_pos[0]] = 0  # mask test links

train_node_information = None
test_node_information = None
if args.use_embedding:
    train_embeddings = generate_node2vec_embeddings(Atrain, args.embedding_dim, True, train_neg) #?
    train_node_information = train_embeddings
    test_embeddings = generate_node2vec_embeddings(Atest, args.embedding_dim, True, test_neg) #?
    test_node_information = test_embeddings
if args.use_attribute and trainAttributes is not None: 
    if train_node_information is not None:
        train_node_information = np.concatenate([train_node_information, trainAttributes], axis=1)
        test_node_information = np.concatenate([test_node_information, testAttributes], axis=1)
    else:
        train_node_information = trainAttributes
        test_node_information = testAttributes

train_graphs, test_graphs, train_labels, test_labels = extractLinks2subgraphsSVM(Atrain, Atest, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information, test_node_information)


# For training on 2 datasets, test on 1 dataset
if args.traindata_name2 is not None:
    trainNet2_ori = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.traindata_name2)))
    trainGroup2 = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.traindata_name2)))
    trainNet2 = np.load(args.file_dir+'/data/dream/'+args.traindata_name2+'_pmatrix_'+str(args.pearson_net)+'.npy').tolist()
    # trainNet2 = np.load(args.file_dir+'/data/dream/'+args.traindata_name2+'_mmatrix_'+str(args.pearson_net)+'.npy').tolist()
    allx2 =trainGroup2.toarray().astype('float32')

    #deal with the features:
    trainAttributes2 = genenet_attribute(allx2,dreamTFdict[args.traindata_name2])
    train_pos2, train_neg2, _, _ = sample_neg(trainNet2_ori, 0.0, max_train_num=args.max_train_num)

    Atrain2 = trainNet2.copy()  # the observed network

    train_node_information2 = None
    if args.use_embedding:
        train_embeddings2 = generate_node2vec_embeddings(Atrain2, args.embedding_dim, True, train_neg2) #?
        train_node_information2 = train_embeddings2
    if args.use_attribute and trainAttributes2 is not None: 
        if train_node_information2 is not None:
            train_node_information2 = np.concatenate([train_node_information2, trainAttributes2], axis=1)
        else:
            train_node_information2 = trainAttributes2

    train_graphs2, _, train_labels2, _ = links2subgraphsTranSVM(Atrain2, Atest, train_pos2, train_neg2, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information2, test_node_information)
    train_graphs = train_graphs + train_graphs2
    train_labels = train_labels + train_labels2
    if train_node_information is not None:
        train_node_information = np.concatenate([train_node_information, train_node_information2], axis=0)
print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

X=np.asarray(train_graphs)
y=np.asarray(train_labels)
testx=np.asarray(test_graphs)
true_y=np.asarray(test_labels)

# clf = LinearSVC()
clf = svm.SVC(gamma='scale')
# clf = svm.SVC()
clf.fit(X, y)
pred=clf.predict(testx)
y_score=clf.decision_function(testx)
print(classification_report(true_y, pred))
tn, fp, fn, tp = confusion_matrix(true_y,pred).ravel()
print(str(tp)+"\t"+str(fp)+"\t"+str(tn)+"\t"+str(fn))

np.save('svm_true_y_34_all.npy',true_y)
np.save('svm_y_score_34_all.npy',y_score)

# precision, recall, _ = precision_recall_curve(true_y, y_score)
# # plot no skill
# plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# # plot the precision-recall curve for the model
# plt.plot(recall, precision, marker='.')
# # show the plot
# #plt.show()
# plt.savefig('SVM_34_e.png')


#randomforest
rf = RandomForestClassifier()
rf.fit(X, y)
pred=rf.predict(testx)
y_score=rf.predict_proba(testx)
print(classification_report(true_y, pred))
tn, fp, fn, tp = confusion_matrix(true_y,pred).ravel()
print(str(tp)+"\t"+str(fp)+"\t"+str(tn)+"\t"+str(fn))
np.save('rf_true_y_34_all.npy',true_y)
np.save('rf_y_score_34_all.npy',y_score)

