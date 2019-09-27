from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
import networkx as nx

pos_graphs_labels = np.load("pos_graphs_labels.npy")
pos_graphs_features = np.load("pos_graphs_features.npy")
neg_graphs_labels = np.load("neg_graphs_labels.npy")
neg_graphs_features = np.load("neg_graphs_features_zscore.npy")

#2055 vs 1448674

def basic_info(graphs_labels=neg_graphs_labels, graphs_features=neg_graphs_features, outfilename='summary_neg.txt'):
    #example:
    # #[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # labels = pos_graphs_labels[0]
    # #[99,806]
    # node_features = pos_graphs_features[0].shape
    outList = []

    graphSizeDict = {}
    labelSizeDict = {}
    labelMaxSizeDict = {}
    # for i in np.arange(len(pos_graphs_labels)):
    #     labels = pos_graphs_labels[i]
    #     graphSize = pos_graphs_features[i].shape[0] 
    for i in np.arange(len(graphs_labels)):
        labels = graphs_labels[i]
        graphSize = graphs_features[i].shape[0]

        labelSize = len(np.nonzero(labels)[0])

        labelMaxSize = max(np.nonzero(labels)[0])
        outList.append(str(graphSize)+'\t'+str(labelSize)+'\t'+str(labelMaxSize))

        if graphSize in graphSizeDict:
            graphSizeDict[graphSize] = graphSizeDict[graphSize] + 1
        else:
            graphSizeDict[graphSize] = 1

        if labelSize in labelSizeDict:
            labelSizeDict[labelSize] = labelSizeDict[labelSize] + 1
        else:
            labelSizeDict[labelSize] = 1

        if labelMaxSize in labelMaxSizeDict:
            labelMaxSizeDict[labelMaxSize] = labelMaxSizeDict[labelMaxSize] + 1
        else:
            labelMaxSizeDict[labelMaxSize] = 1


    for key in sorted(graphSizeDict):
        print(str(key)+"\t"+str(graphSizeDict[key]))

    print("*****")
    for key in sorted(labelSizeDict):
        print(str(key)+"\t"+str(labelSizeDict[key]))

    print("#####")
    for key in sorted(labelMaxSizeDict):
        print(str(key)+"\t"+str(labelMaxSizeDict[key]))

    with open(outfilename, 'w') as f:
        for item in outList:
            f.write('%s\n' %item)




# def summary_motif(graphs_labels=neg_graphs_labels, graphs_features=neg_graphs_features):



basic_info(graphs_labels=pos_graphs_labels, graphs_features=pos_graphs_features, outfilename='summary_pos.txt')
# basic_info(graphs_labels=neg_graphs_labels, graphs_features=neg_graphs_features, outfilename='summary_neg.txt')