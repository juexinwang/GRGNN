from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
import networkx as nx

#2055 vs 1448674
pos_graphs_labels = np.load("pos_graphs_labels.npy")
pos_graphs_features = np.load("pos_graphs_features_zscore.npy")
neg_graphs_labels = np.load("neg_graphs_labels.npy")
neg_graphs_features = np.load("neg_graphs_features_zscore.npy")

class GraphInfo(object):
    def __init__(self, index, labels, features):
        self.index = index
        self.labels = labels
        self.features = features
        
        #generate information
        self.exEnd1 = features[0,0]
        self.exEnd2 = features[1,0]
        self.exMeanNeighbors = np.average(features[2:,0])
        self.exMaxNeighbors = np.max(features[2:,0])
        self.exMinNeighbors = np.min(features[2:,0])
        
        self.numNeighbor = features.shape[0]-2

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
            self.exMeanNeighborTF = np.average(neighborTF[:,0])
            self.exMaxNeighborTF = np.max(neighborTF[:,0])
            self.exMinNeighborTF = np.min(neighborTF[:,0])

            self.numNeighborTF = neighborTF.shape[0]

        if not len(neighborTarget)==0:
            self.exMeanNeighborTarget = np.average(neighborTarget[:,0])
            self.exMaxNeighborTarget = np.max(neighborTarget[:,0])
            self.exMinNeighborTarget = np.min(neighborTarget[:,0])

            self.numNeighborTarget = neighborTarget.shape[0]

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
            self.exMeanNeighborTFBypass = np.average(neighborTFBypass[:,0])
            self.exMaxNeighborTFBypass = np.max(neighborTFBypass[:,0])
            self.exMinNeighborTFBypass = np.min(neighborTFBypass[:,0])

            self.numNeighborTFBypass = neighborTFBypass.shape[0]

        if not len(neighborTargetBypass)==0:
            self.exMeanNeighborTargetBypass = np.average(neighborTargetBypass[:,0])
            self.exMaxNeighborTargetBypass = np.max(neighborTargetBypass[:,0])
            self.exMinNeighborTargetBypass = np.min(neighborTargetBypass[:,0])

            self.numNeighborTargetBypass = neighborTargetBypass.shape[0]

        if not len(neighborTFBypassNo)==0:
            self.exMeanNeighborTFBypassNo = np.average(neighborTFBypassNo[:,0])
            self.exMaxNeighborTFBypassNo = np.max(neighborTFBypassNo[:,0])
            self.exMinNeighborTFBypassNo = np.min(neighborTFBypassNo[:,0])

            self.numNeighborTFBypassNo = neighborTFBypassNo.shape[0]

        if not len(neighborTargetBypassNo)==0:
            self.exMeanNeighborTargetBypassNo = np.average(neighborTargetBypassNo[:,0])
            self.exMaxNeighborTargetBypassNo = np.max(neighborTargetBypassNo[:,0])
            self.exMinNeighborTargetBypassNo = np.min(neighborTargetBypassNo[:,0])

            self.numNeighborTargetBypassNo = neighborTargetBypassNo.shape[0]

#condition1: TF-TF/TF-Target
def condition1(gi):
    #TF-TF
    if gi.features[0,-1]==1.0 and gi.features[1,-1]==1.0 :
        result = True   
    #TF-Target
    elif gi.features[0,-1]==1.0 and gi.features[1,-1]==0.0 :
        result = False
    return result

#condition2: For TF-Target: end1>end2/end1<end2
def condition2(gi):
    #end1>end2
    if gi.exEnd1 > gi.exEnd2 :
        result = True   
    #end1<end2
    else:
        result = False
    return result

#condition3: if have neighbors: MeanNeighbors>end1/MeanNeighbors<end1
def condition3(gi):
    #MeanNeighbors>end1
    if gi.exMeanNeighbors > gi.exEnd1 :
        result = True   
    #MeanNeighbors<end1
    else:
        result = False
    return result

#condition4: if have neighbors: MeanNeighbors>end2/MeanNeighbors<end2
def condition4(gi):
    #MeanNeighbors>end2
    if gi.exMeanNeighbors > gi.exEnd2 :
        result = True   
    #MeanNeighbors<end2
    else:
        result = False
    return result

#condition5: if neighbors have TF: exMeanNeighborTF>end1/exMeanNeighborTF<end1
def condition5(gi):
    #exMeanNeighborTF>end1
    if gi.exMeanNeighborTF > gi.exEnd1 :
        result = True   
    #exMeanNeighborTF<end1
    else:
        result = False
    return result

#condition6: if neighbors have TF: exMeanNeighborTF>end2/exMeanNeighborTF<end2
def condition6(gi):
    #exMeanNeighborTF>end2
    if gi.exMeanNeighborTF > gi.exEnd2 :
        result = True   
    #exMeanNeighborTF<end2
    else:
        result = False
    return result

#condition7: if neighbors have Target: exMeanNeighborTarget>end1/exMeanNeighborTarget<end1
def condition7(gi):
    #exMeanNeighborTarget>end1
    if gi.exMeanNeighborTarget > gi.exEnd1 :
        result = True   
    #exMeanNeighborTarget<end1
    else:
        result = False
    return result

#condition8: if neighbors have Target: exMeanNeighborTarget>end2/exMeanNeighborTarget<end2
def condition8(gi):
    #exMeanNeighborTarget>end2
    if gi.exMeanNeighborTarget > gi.exEnd2 :
        result = True   
    #exMeanNeighborTarget<end2
    else:
        result = False
    return result

#condition9: if neighbors have TF: if have bypass: exMeanNeighborTFBypass>end1/exMeanNeighborTFBypass<end1
def condition9(gi):
    #exMeanNeighborTFBypass>end1
    if gi.exMeanNeighborTFBypass > gi.exEnd1 :
        result = True   
    #exMeanNeighborTFBypass<end1
    else:
        result = False
    return result

#condition10: if neighbors have TF: if have bypass: exMeanNeighborTFBypass>end2/exMeanNeighborTFBypass<end2
def condition10(gi):
    #exMeanNeighborTFBypass>end2
    if gi.exMeanNeighborTFBypass > gi.exEnd2 :
        result = True   
    #exMeanNeighborTFBypass<end2
    else:
        result = False
    return result

#condition11: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass>end1/exMeanNeighborTargetBypass<end1
def condition11(gi):
    #exMeanNeighborTargetBypass>end1
    if gi.exMeanNeighborTargetBypass > gi.exEnd1 :
        result = True   
    #exMeanNeighborTargetBypass<end1
    else:
        result = False
    return result

#condition12: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass>end2/exMeanNeighborTargetBypass<end2
def condition12(gi):
    #exMeanNeighborTargetBypass>end2
    if gi.exMeanNeighborTargetBypass > gi.exEnd2 :
        result = True   
    #exMeanNeighborTargetBypass<end2
    else:
        result = False
    return result


#condition13: if neighbors have TF: if have no bypass: exMeanNeighborTFBypassNo>end1/exMeanNeighborTFBypassNo<end1
def condition13(gi):
    #exMeanNeighborTFBypassNo>end1
    if gi.exMeanNeighborTFBypassNo > gi.exEnd1 :
        result = True   
    #exMeanNeighborTFBypassNo<end1
    else:
        result = False
    return result

#condition14: if neighbors have TF: if have no bypass: exMeanNeighborTFBypassNo>end2/exMeanNeighborTFBypassNo<end2
def condition14(gi):
    #exMeanNeighborTFBypassNo>end2
    if gi.exMeanNeighborTFBypassNo > gi.exEnd2 :
        result = True   
    #exMeanNeighborTFBypassNo<end2
    else:
        result = False
    return result

#condition15: if neighbors have Target: if have no bypass: exMeanNeighborTargetBypassNo>end1/exMeanNeighborTargetBypassNo<end1
def condition15(gi):
    #exMeanNeighborTargetBypassNo>end1
    if gi.exMeanNeighborTargetBypassNo > gi.exEnd1 :
        result = True   
    #exMeanNeighborTargetBypassNo<end1
    else:
        result = False
    return result

#condition16: if neighbors have Target: if have no bypass: exMeanNeighborTargetBypassNo>end2/exMeanNeighborTargetBypassNo<end2
def condition16(gi):
    #exMeanNeighborTargetBypassNo>end2
    if gi.exMeanNeighborTargetBypassNo > gi.exEnd2 :
        result = True   
    #exMeanNeighborTargetBypassNo<end2
    else:
        result = False
    return result




#posFlag: True for positive, False for negative
def generate_information(graphs_labels=pos_graphs_labels, graphs_features=pos_graphs_features):

    giList=[]
    for i in np.arange(len(graphs_labels)):
        labels = graphs_labels[i]
        features = graphs_features[i]
        giList.append(GraphInfo(i,labels,features))

    return giList

posGiList = generate_information(graphs_labels=pos_graphs_labels, graphs_features=pos_graphs_features)
# negGiList = generate_information(graphs_labels=neg_graphs_labels, graphs_features=neg_graphs_features)

num_TT=0
num_TG=0
num_TT_n12=0
num_TT_1n2=0
num_TT_12n=0
num_TG_12=0
num_TG_21=0
num_TG_n12=0
num_TG_1n2=0
num_TG_12n=0
num_TG_n21=0
num_TG_2n1=0
num_TG_21n=0
for gi in posGiList:
    if condition1(gi):
        num_TT = num_TT + 1
        if condition2(gi):
            if condition3(gi):
                num_TT_n12 = num_TT_n12 +1
            else:
                if condition4(gi):
                    num_TT_1n2 = num_TT_1n2 +1
                else:
                    num_TT_12n = num_TT_12n +1
        else:
            if condition4(gi):
                num_TT_n12 = num_TT_n12 +1
            else:
                if condition3(gi):
                    num_TT_1n2 = num_TT_1n2 +1
                else:
                    num_TT_12n = num_TT_12n +1
    else:
        num_TG = num_TG + 1
        if condition2(gi):
            num_TG_12 = num_TG_12 +1
            if condition3(gi):
                num_TG_n12 = num_TG_n12 +1
            else:
                if condition4(gi):
                    num_TG_1n2 = num_TG_1n2 +1
                else:
                    num_TG_12n = num_TG_12n +1
        else:
            num_TG_21 = num_TG_21 +1
            if condition4(gi):
                num_TG_n21 = num_TG_n21 +1
            else:
                if condition3(gi):
                    num_TG_2n1 = num_TG_2n1 +1
                else:
                    num_TG_21n = num_TG_21n +1 




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


# basic_info(graphs_labels=pos_graphs_labels, graphs_features=pos_graphs_features, outfilename='summary_pos.txt')
# basic_info(graphs_labels=neg_graphs_labels, graphs_features=neg_graphs_features, outfilename='summary_neg.txt')