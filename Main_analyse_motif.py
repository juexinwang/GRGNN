from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
import networkx as nx

pos_graphs_labels = np.load("pos_graphs_labels.npy")
pos_graphs_features = np.load("pos_graphs_features_zscore.npy")
neg_graphs_labels = np.load("neg_graphs_labels.npy")
neg_graphs_features = np.load("neg_graphs_features_zscore.npy")

class GraphInfo(object):
    def __init__(self, index, labels, features):
        self.index = index
        self.labels = labels
        self.features = features 

        


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

def condition1(ele1,ele2,category):
    result=False


    return result

#posFlag: True for positive, False for negative
def summary_motif(graphs_labels=pos_graphs_labels, graphs_features=pos_graphs_features, posFlag=True):

    # TF-TF/TF-Target
    for i in np.arange(len(graphs_labels)):
        labels = graphs_labels[i]
        features = graphs_features[i]

        exEnd1 = features[0,0]
        exEnd2 = features[1,0]
        exMeanNeighbors = np.average(features[2:,0])
        exMaxNeighbors = np.max(features[2:,0])
        exMinNeighbors = np.min(features[2:,0])

        # neighbors are TF or not, their number and expression
        neighborTF =[]
        neighborTarget = []
        neighborLabelTF = []
        neighborLabelTarget = []
        for j in np.arange(features.shape[0]):
            if features[j+2,-1]==1.0:
                neighborTF.append(features[j+2,:])
                neighborLabelTF.append(labels[j+2])
            else:
                neighborTarget.append(features[j+2,:])
                neighborLabelTarget.append(labels[j+2])
        neighborTF=np.asarray(neighborTF)
        neighborTarget=np.asarray(neighborTarget)
        neighborLabelTF=np.asarray(neighborTF)
        neighborLabelTarget=np.asarray(neighborTarget)
        
        if not len(neighborTF)==0:
            exMeanNeighborTF = np.average(neighborTF[:,0])
            exMaxNeighborTF = np.max(neighborTF[:,0])
            exMinNeighborTF = np.min(neighborTF[:,0])

        if not len(neighborTarget)==0:
            exMeanNeighborTarget = np.average(neighborTarget[:,0])
            exMaxNeighborTarget = np.max(neighborTarget[:,0])
            exMinNeighborTarget = np.min(neighborTarget[:,0])

        # neighbors have bypass or not, the number of bypass and their expression
        neighborTFBypass =[]
        neighborTFBypassNo =[]
        neighborTargetBypass = []
        neighborTargetBypassNo = []
        neighborLabelTFBypass = []
        neighborLabelTFBypassNo = []
        neighborLabelTargetBypass = []
        neighborLabelTargetBypassNo = []
        for j in np.arange(len(neighborLabelTF)):
            #bypass
            if neighborLabelTF[j]>0:
                neighborTFBypass.append(neighborTF[j,:])
                neighborLabelTFBypass.append(neighborLabelTF[j])
            # nobypass
            else:
                neighborTFBypassNo.append(neighborTF[j,:])
                neighborLabelTFBypassNo.append(neighborLabelTF[j])
        
        for j in np.arange(len(neighborLabelTarget)):
            #bypass
            if neighborLabelTarget[j]>0:
                neighborTargetBypass.append(neighborTarget[j,:])
                neighborLabelTargetBypass.append(neighborLabelTarget[j])
            # nobypass
            else:
                neighborTargetBypassNo.append(neighborTarget[j,:])
                neighborLabelTargetBypassNo.append(neighborLabelTarget[j])

        neighborTFBypass = np.asarray(neighborTFBypass)
        neighborTFBypassNo = np.asarray(neighborTFBypassNo)
        neighborTargetBypass = np.asarray(neighborTargetBypass)
        neighborTargetBypassNo = np.asarray(neighborTargetBypassNo)
        neighborLabelTFBypass = np.asarray(neighborLabelTFBypass)
        neighborLabelTFBypassNo = np.asarray(neighborLabelTFBypassNo)
        neighborLabelTargetBypass = np.asarray(neighborLabelTargetBypass)
        neighborLabelTargetBypassNo = np.asarray(neighborLabelTargetBypassNo)

        if not len(neighborTFBypass)==0:
            exMeanNeighborTFBypass = np.average(neighborTFBypass[:,0])
            exMaxNeighborTFBypass = np.max(neighborTFBypass[:,0])
            exMinNeighborTFBypass = np.min(neighborTFBypass[:,0])

        if not len(neighborTargetBypass)==0:
            exMeanNeighborTargetBypass = np.average(neighborTargetBypass[:,0])
            exMaxNeighborTargetBypass = np.max(neighborTargetBypass[:,0])
            exMinNeighborTargetBypass = np.min(neighborTargetBypass[:,0])

        if not len(neighborTFBypassNo)==0:
            exMeanNeighborTFBypassNo = np.average(neighborTFBypassNo[:,0])
            exMaxNeighborTFBypassNo = np.max(neighborTFBypassNo[:,0])
            exMinNeighborTFBypassNo = np.min(neighborTFBypassNo[:,0])

        if not len(neighborTargetBypassNo)==0:
            exMeanNeighborTargetBypassNo = np.average(neighborTargetBypassNo[:,0])
            exMaxNeighborTargetBypassNo = np.max(neighborTargetBypassNo[:,0])
            exMinNeighborTargetBypassNo = np.min(neighborTargetBypassNo[:,0])

        # All the numbers
        numNeighborTF = neighborTF.shape[0]
        numNeighborTarget = neighborTarget.shape[0]
        numNeighbor = features.shape[0]-2

        numNeighborTFBypass = neighborTFBypass.shape[0]
        numNeighborTargetBypass = neighborTargetBypass.shape[0]

        numNeighborTFBypassNo = neighborTFBypassNo.shape[0]
        numNeighborTargetBypassNo = neighborTargetBypassNo.shape[0]



    # #TF-TF
    # if features[0,-1]==1.0 and features[1,-1]==1.0 :
        
    # #TF-Target
    # elif features[0,-1]==1.0 and features[1,-1]==0.0 :
    #     exNeighbors = np.average(features[2:,0])
    # #others, should be wrong
    # else:
    #     print("Something Wrong!")

    #Possible Rules:
    if exEnd1 < exEnd2 and 


    return



basic_info(graphs_labels=pos_graphs_labels, graphs_features=pos_graphs_features, outfilename='summary_pos.txt')
# basic_info(graphs_labels=neg_graphs_labels, graphs_features=neg_graphs_features, outfilename='summary_neg.txt')