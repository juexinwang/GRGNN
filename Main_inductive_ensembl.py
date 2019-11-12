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


parser = argparse.ArgumentParser(description='Gene Regulatory Graph Neural Network in ensemble')
# general settings
parser.add_argument('--traindata-name', default='dream3', help='train network name')
parser.add_argument('--traindata-name2', default=None, help='also train another network')
parser.add_argument('--testdata-name', default='dream4', help='test network name')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--training-ratio', type=float, default=1.0,
                    help='ratio of used training set')
parser.add_argument('--neighbors-ratio', type=float, default=1.0,
                    help='ratio of neighbors used')
parser.add_argument('--nonezerolabel-flag', default=True,
                    help='whether only use nonezerolabel flag')
# Pearson correlation
parser.add_argument('--embedding-dim', type=int, default=128,
                    help='embedding dimmension')
parser.add_argument('--pearson_net', type=float, default=0.8, #1
                    help='pearson correlation as the network')
parser.add_argument('--mutual_net', type=int, default=3, #3
                    help='mutual information as the network')
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


# Inductive learning
# For 1vs 1
if args.traindata_name is not None:
    trainNet_ori = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.traindata_name)))
    trainGroup = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.traindata_name)))
    # trainNet_agent0 = np.load(args.file_dir+'/data/dream/'+args.traindata_name+'_pmatrix_'+str(args.pearson_net)+'.npy').tolist()
    # trainNet_agent1 = np.load(args.file_dir+'/data/dream/'+args.traindata_name+'_mmatrix_'+str(args.mutual_net)+'.npy').tolist()
    trainNet_agent0 = np.load(args.file_dir+'/data/dream/'+args.traindata_name+'_rmatrix_0.003.npy').tolist()
    trainNet_agent1 = np.load(args.file_dir+'/data/dream/'+args.traindata_name+'_rmatrix_0.003.npy').tolist()
    
    allx =trainGroup.toarray().astype('float32')
    #deal with the features:
    trainAttributes = genenet_attribute(allx,dreamTFdict[args.traindata_name])   

    testNet_ori = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.testdata_name)))
    testGroup = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.testdata_name)))
    # testNet_agent0 = np.load(args.file_dir+'/data/dream/'+args.testdata_name+'_pmatrix_'+str(args.pearson_net)+'.npy').tolist()
    # testNet_agent1 = np.load(args.file_dir+'/data/dream/'+args.testdata_name+'_mmatrix_'+str(args.mutual_net)+'.npy').tolist()
    testNet_agent0 = np.load(args.file_dir+'/data/dream/'+args.testdata_name+'_rmatrix_0.003.npy').tolist()
    testNet_agent1 = np.load(args.file_dir+'/data/dream/'+args.testdata_name+'_rmatrix_0.003.npy').tolist()
    
    allxt =testGroup.toarray().astype('float32')
    #deal with the features:
    testAttributes = genenet_attribute(allxt,dreamTFdict[args.testdata_name])

    train_pos, train_neg, _, _ = sample_neg_TF(trainNet_ori, 0.0, TF_num=dreamTFdict[args.traindata_name], max_train_num=args.max_train_num)
    use_pos_size = math.floor(len(train_pos[0])*args.training_ratio)
    use_neg_size = math.floor(len(train_neg[0])*args.training_ratio)
    train_pos=(train_pos[0][:use_pos_size],train_pos[1][:use_pos_size])
    train_neg=(train_neg[0][:use_neg_size],train_neg[1][:use_neg_size])
    _, _, test_pos, test_neg = sample_neg_TF(testNet_ori, 1.0, TF_num=dreamTFdict[args.testdata_name], max_train_num=args.max_train_num)
    # test_pos, test_neg = sample_neg_all_TF(testNet_ori, TF_num=dreamTFdict[args.testdata_name])



'''Train and apply classifier'''
Atrain_agent0 = trainNet_agent0.copy()  # the observed network
Atrain_agent1 = trainNet_agent1.copy()
Atest_agent0 = testNet_agent0.copy()  # the observed network
Atest_agent1 = testNet_agent1.copy()
Atest_agent0[test_pos[0], test_pos[1]] = 0  # mask test links
Atest_agent0[test_pos[1], test_pos[0]] = 0  # mask test links
Atest_agent1[test_pos[0], test_pos[1]] = 0  # mask test links
Atest_agent1[test_pos[1], test_pos[0]] = 0  # mask test links

# train_node_information = None
# test_node_information = None
if args.use_embedding:
    train_embeddings_agent0 = generate_node2vec_embeddings(Atrain_agent0, args.embedding_dim, True, train_neg) #?
    train_node_information_agent0 = train_embeddings_agent0
    test_embeddings_agent0 = generate_node2vec_embeddings(Atest_agent0, args.embedding_dim, True, test_neg) #?
    test_node_information_agent0 = test_embeddings_agent0

    train_embeddings_agent1 = generate_node2vec_embeddings(Atrain_agent1, args.embedding_dim, True, train_neg) #?
    train_node_information_agent1 = train_embeddings_agent1
    test_embeddings_agent1 = generate_node2vec_embeddings(Atest_agent1, args.embedding_dim, True, test_neg) #?
    test_node_information_agent1 = test_embeddings_agent1
if args.use_attribute and trainAttributes is not None: 
    if args.use_embedding:
        train_node_information_agent0 = np.concatenate([train_node_information_agent0, trainAttributes], axis=1)
        test_node_information_agent0 = np.concatenate([test_node_information_agent0, testAttributes], axis=1)

        train_node_information_agent1 = np.concatenate([train_node_information_agent1, trainAttributes], axis=1)
        test_node_information_agent1 = np.concatenate([test_node_information_agent1, testAttributes], axis=1)
    else:
        train_node_information_agent0 = trainAttributes
        test_node_information_agent0 = testAttributes

        train_node_information_agent1 = trainAttributes
        test_node_information_agent1 = testAttributes

# Original
# train_graphs_agent0, test_graphs_agent0, max_n_label_agent0 = extractLinks2subgraphs(Atrain_agent0, Atest_agent0, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information_agent0, test_node_information_agent0)
# train_graphs_agent1, test_graphs_agent1, max_n_label_agent1 = extractLinks2subgraphs(Atrain_agent1, Atest_agent1, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information_agent1, test_node_information_agent1)

# ratio
train_graphs_agent0, test_graphs_agent0, max_n_label_agent0 = extractLinks2subgraphsRatio(Atrain_agent0, Atest_agent0, train_pos, train_neg, test_pos, test_neg, args.neighbors_ratio, args.nonezerolabel_flag, args.hop, args.max_nodes_per_hop, train_node_information_agent0, test_node_information_agent0)
train_graphs_agent1, test_graphs_agent1, max_n_label_agent1 = extractLinks2subgraphsRatio(Atrain_agent1, Atest_agent1, train_pos, train_neg, test_pos, test_neg, args.neighbors_ratio, args.nonezerolabel_flag, args.hop, args.max_nodes_per_hop, train_node_information_agent1, test_node_information_agent1)


# For 2 vs 1
if args.traindata_name2 is not None:
    trainNet2_ori = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.csc'.format(args.traindata_name2)))
    trainGroup2 = np.load(os.path.join(args.file_dir, 'data/dream/ind.{}.allx'.format(args.traindata_name2)))
    trainNet2_agent0 = np.load(args.file_dir+'/data/dream/'+args.traindata_name2+'_pmatrix_'+str(args.pearson_net)+'.npy').tolist()
    trainNet2_agent1 = np.load(args.file_dir+'/data/dream/'+args.traindata_name2+'_mmatrix_'+str(args.mutual_net)+'.npy').tolist()
    allx2 =trainGroup2.toarray().astype('float32')

    #deal with the features:
    trainAttributes2 = genenet_attribute(allx2,dreamTFdict[args.traindata_name2])
    train_pos2, train_neg2, _, _ = sample_neg_TF(trainNet2_ori, 0.0, TF_num=dreamTFdict[args.traindata_name2], max_train_num=args.max_train_num,semi_pool_fold=args.semi_pool_fold)

    Atrain2_agent0 = trainNet2_agent0.copy()  # the observed network
    Atrain2_agent1 = trainNet2_agent1.copy()

    train_node_information2 = None
    if args.use_embedding:
        train_embeddings2_agent0 = generate_node2vec_embeddings(Atrain2_agent0, args.embedding_dim, True, train_neg2) #?
        train_node_information2_agent0 = train_embeddings2_agent0

        train_embeddings2_agent1 = generate_node2vec_embeddings(Atrain2_agent1, args.embedding_dim, True, train_neg2) #?
        train_node_information2_agent1 = train_embeddings2_agent1
    if args.use_attribute and trainAttributes2 is not None: 
        if args.use_embedding:
            train_node_information2_agent0 = np.concatenate([train_node_information2_agent0, trainAttributes2], axis=1)
            train_node_information2_agent1 = np.concatenate([train_node_information2_agent1, trainAttributes2], axis=1)
        else:
            train_node_information2_agent0 = trainAttributes2
            train_node_information2_agent1 = trainAttributes2

    train_graphs2_agent0, _, max_n_label_agent0 = extractLinks2subgraphs(Atrain2_agent0, Atest_agent0, train_pos2, train_neg2, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information2_agent0, test_node_information_agent0)
    train_graphs_agent0 = train_graphs_agent0 + train_graphs2_agent0
    train_graphs2_agent1, _, max_n_label_agent1 = extractLinks2subgraphs(Atrain2_agent1, Atest_agent1, train_pos2, train_neg2, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information2_agent1, test_node_information_agent1)
    train_graphs_agent1 = train_graphs_agent1 + train_graphs2_agent1
    if args.use_embedding:
        train_node_information_agent0 = np.concatenate([train_node_information_agent0, train_node_information2_agent0], axis=0)
        train_node_information_agent1 = np.concatenate([train_node_information_agent1, train_node_information2_agent1], axis=0)
print('# train: %d, # test: %d' % (len(train_graphs_agent0), len(test_graphs_agent0)))


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

_, _, test_neg_agent0,  _,test_prob_agent0 =DGCNN_classifer(train_graphs_agent0, test_graphs_agent0, train_node_information_agent0, max_n_label_agent0, set_epoch = 50, eval_flag=True)

_, _, test_neg_agent1,  _,test_prob_agent1 =DGCNN_classifer(train_graphs_agent1, test_graphs_agent1, train_node_information_agent1, max_n_label_agent1, set_epoch = 50, eval_flag=True)

dic_agent0={}
for i in test_neg_agent0:
    dic_agent0[i]=0
dic_agent1={}
for i in test_neg_agent1:
    dic_agent1[i]=0
bothwrong = 0
corrected = 0
uncorrected = 0
count = 0

tp0=0
tp1=0
tn0=0
tn1=0
tp=0
tn=0
testpos_size = len(test_pos[0])
for i in np.arange(len(test_prob_agent0)):
    if i<testpos_size: #positive part
        if i in dic_agent0 or i in dic_agent1:
            if test_prob_agent0[i]*test_prob_agent1[i]>0:
                # both wrong
                bothwrong = bothwrong + 1
            else:
                if abs(test_prob_agent0[i])>abs(test_prob_agent1[i]):
                    if i in dic_agent0 and i not in dic_agent1:
                        uncorrected = uncorrected +1
                        tp1 = tp1 + 1
                    else:
                        corrected = corrected +1
                        count = count +1
                        tp = tp +1
                        tp0 = tp0 + 1 
                else:
                    if i in dic_agent0 and i not in dic_agent1:
                        corrected = corrected +1
                        count = count +1
                        tp = tp +1
                        tp1 = tp1 + 1
                    else:
                        uncorrected = uncorrected +1  
                        tp0 = tp0 + 1                  
        else:
            count = count +1
            tp = tp +1
            tp0 = tp0 + 1
            tp1 = tp1 + 1
    else: #negative part
        if i in dic_agent0 or i in dic_agent1:
            if test_prob_agent0[i]*test_prob_agent1[i]>0:
                # both wrong
                bothwrong = bothwrong + 1
            else:
                if abs(test_prob_agent0[i])>abs(test_prob_agent1[i]):
                    if i in dic_agent0 and i not in dic_agent1:
                        uncorrected = uncorrected +1
                        tn1 = tn1 + 1
                    else:
                        corrected = corrected +1
                        count = count +1 
                        tn = tn+1
                        tn0 = tn0 + 1
                else:
                    if i in dic_agent0 and i not in dic_agent1:
                        corrected = corrected +1
                        count = count +1
                        tn = tn+1
                        tn1 = tn1 + 1
                    else:
                        uncorrected = uncorrected +1  
                        tn0 = tn0 + 1                  
        else:
            count = count +1
            tn = tn +1 
            tn0 = tn0 + 1
            tn1 = tn1 + 1

print("Both agents right: "+str(count))
print("Both agents wrong: "+str(bothwrong))
print("Corrected by Ensembl: "+str(corrected))
print("Not corrected by Ensembl: "+str(uncorrected))

allstr = str(float((tp+tn)/len(test_graphs_agent0)))+"\t"+str(tp)+"\t"+str(len(test_pos[0])-tp)+"\t"+str(tn)+"\t"+str(len(test_neg[0])-tn)
agent0_str = str(float((tp0+tn0)/len(test_graphs_agent0)))+"\t"+str(tp0)+"\t"+str(len(test_pos[0])-tp0)+"\t"+str(tn0)+"\t"+str(len(test_neg[0])-tn0)
agent1_str = str(float((tp1+tn1)/len(test_graphs_agent0)))+"\t"+str(tp1)+"\t"+str(len(test_pos[0])-tp1)+"\t"+str(tn1)+"\t"+str(len(test_neg[0])-tn1)
result = str(float(count/len(test_graphs_agent0)))
print("Ensemble:Accuracy tp fn tn fp")
print(allstr+"\n")
print("Agent0:Accuracy tp fn tn fp")
print(agent0_str+"\n")   
print("Agent1:Accuracy tp fn tn fp")
print(agent1_str+"\n") 

with open('acc_result.txt', 'a+') as f:
    f.write(allstr+"\t"+agent0_str+"\t"+agent1_str + '\n')




