from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
import networkx as nx
# cur_dir = os.path.dirname(os.path.realpath(__file__))
cur_dir = '/home/wangjue/myprojects/GRGNN'
sys.path.append('%s/software/pytorch_DGCNN' % cur_dir)
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph
import node2vec

pos_graphs = np.load("pos_graphs.npy")

#example:
# #[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# labels = pos_graphs[0].node_tags
# #[99,806]
# node_features = pos_graphs[0].node_features.shape
outList = []

graphSizeDict = {}
labelSizeDict = {}
labelMaxSizeDict = {}
for graph in pos_graphs:
    labels = graph.node_tags
    graphSize = graph.node_features.shape[0] 

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

with open('summary.txt', 'w') as f:
    for item in outList:
        f.write('%s\n' %item)