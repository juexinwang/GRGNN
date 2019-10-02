from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
import networkx as nx

#3: ecoli
#2055 vs 1448674
#4: yeast
#3935 vs 1921804
pos_graphs_labels = np.load("pos_graphs_labels.npy")
pos_graphs_features = np.load("pos_graphs_features_mean.npy")
neg_graphs_labels = np.load("neg_graphs_labels.npy")
neg_graphs_features = np.load("neg_graphs_features_mean.npy")

ph1=0
ph2=0
ph3=0

class GraphInfo(object):
    def __init__(self, index, labels, features):
        self.index = index
        self.labels = labels
        self.features = features
        
        #generate information
        self.exEnd1 = features[0,0]
        self.exEnd2 = features[1,0]
        if len(features)>2:
            self.exMeanNeighbors = np.average(features[2:,0])
            self.exMaxNeighbors = np.max(features[2:,0])
            self.exMinNeighbors = np.min(features[2:,0])
        
        self.numNeighbor = features.shape[0]-2

        # neighbors are TF or not, their number and expression
        neighborTF =[]
        neighborTarget = []
        neighborLabelTF = []
        neighborLabelTarget = []
        for j in np.arange(features.shape[0]-2):
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

        self.neighborTF=neighborTF
        self.neighborTarget=neighborTarget
        self.neighborLabelTF=neighborTF
        self.neighborLabelTarget=neighborTarget
        
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

        neighborTFBypass_2 =[]
        neighborTFBypass_3 =[]
        neighborTFBypass_H =[]
        neighborTargetBypass_2 = []
        neighborTargetBypass_3 = []
        neighborTargetBypass_H = []
        neighborLabelTFBypass_2 = []
        neighborLabelTFBypass_3 = []
        neighborLabelTFBypass_H = []
        neighborLabelTargetBypass_2 = []
        neighborLabelTargetBypass_3 = []
        neighborLabelTargetBypass_H = []

        for j in np.arange(len(neighborLabelTF)):
            #bypass
            if neighborLabelTF[j,1]>0:
                neighborTFBypass.append(neighborTF[j,:])
                neighborLabelTFBypass.append(neighborLabelTF[j])
                if neighborLabelTF[j,1] == 2:
                    neighborTFBypass_2.append(neighborTF[j,:])
                    neighborLabelTFBypass_2.append(neighborLabelTF[j])
                elif neighborLabelTF[j,1] == 3:
                    neighborTFBypass_3.append(neighborTF[j,:])
                    neighborLabelTFBypass_3.append(neighborLabelTF[j])
                elif neighborLabelTF[j,1] >3:
                    neighborTFBypass_H.append(neighborTF[j,:])
                    neighborLabelTFBypass_H.append(neighborLabelTF[j]) 
            # nobypass
            else:
                neighborTFBypassNo.append(neighborTF[j,:])
                neighborLabelTFBypassNo.append(neighborLabelTF[j])
        
        for j in np.arange(len(neighborLabelTarget)):
            #bypass
            if neighborLabelTarget[j,1]>0:
                neighborTargetBypass.append(neighborTarget[j,:])
                neighborLabelTargetBypass.append(neighborLabelTarget[j])
                if neighborLabelTarget[j,1] == 2:
                    neighborTargetBypass_2.append(neighborTarget[j,:])
                    neighborLabelTargetBypass_2.append(neighborLabelTarget[j])
                elif neighborLabelTarget[j,1] == 3:
                    neighborTargetBypass_3.append(neighborTarget[j,:])
                    neighborLabelTargetBypass_3.append(neighborLabelTarget[j])
                if neighborLabelTarget[j,1] > 3:
                    neighborTargetBypass_H.append(neighborTarget[j,:])
                    neighborLabelTargetBypass_H.append(neighborLabelTarget[j])   
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

        neighborTFBypass_2 = np.asarray(neighborTFBypass_2)
        neighborTFBypass_3 = np.asarray(neighborTFBypass_3)
        neighborTFBypass_H = np.asarray(neighborTFBypass_H)
        neighborTargetBypass_2 = np.asarray(neighborTargetBypass_2)
        neighborTargetBypass_3 = np.asarray(neighborTargetBypass_3)
        neighborTargetBypass_H = np.asarray(neighborTargetBypass_H)
        neighborLabelTFBypass_2 = np.asarray(neighborLabelTFBypass_2)
        neighborLabelTFBypass_3 = np.asarray(neighborLabelTFBypass_3)
        neighborLabelTFBypass_H = np.asarray(neighborLabelTFBypass_H)
        neighborLabelTargetBypass_2 = np.asarray(neighborLabelTargetBypass_2)
        neighborLabelTargetBypass_3 = np.asarray(neighborLabelTargetBypass_3)
        neighborLabelTargetBypass_H = np.asarray(neighborLabelTargetBypass_H)

        self.neighborTFBypass = neighborTFBypass
        self.neighborTFBypassNo = neighborTFBypassNo
        self.neighborTargetBypass = neighborTargetBypass
        self.neighborTargetBypassNo = neighborTargetBypassNo
        self.neighborLabelTFBypass = neighborLabelTFBypass
        self.neighborLabelTFBypassNo = neighborLabelTFBypassNo
        self.neighborLabelTargetBypass = neighborLabelTargetBypass
        self.neighborLabelTargetBypassNo = neighborLabelTargetBypassNo

        self.neighborTFBypass_2 = neighborTFBypass_2
        self.neighborTFBypass_3 = neighborTFBypass_3
        self.neighborTFBypass_H = neighborTFBypass_H
        self.neighborTargetBypass_2 = neighborTargetBypass_2
        self.neighborTargetBypass_3 = neighborTargetBypass_3
        self.neighborTargetBypass_H = neighborTargetBypass_H
        self.neighborLabelTFBypass_2 = neighborLabelTFBypass_2
        self.neighborLabelTFBypass_3 = neighborLabelTFBypass_3
        self.neighborLabelTFBypass_H = neighborLabelTFBypass_H
        self.neighborLabelTargetBypass_2 = neighborLabelTargetBypass_2
        self.neighborLabelTargetBypass_3 = neighborLabelTargetBypass_3
        self.neighborLabelTargetBypass_H = neighborLabelTargetBypass_H

        if not len(neighborTFBypass)==0:
            self.exMeanNeighborTFBypass = np.average(neighborTFBypass[:,0])
            self.exMaxNeighborTFBypass = np.max(neighborTFBypass[:,0])
            self.exMinNeighborTFBypass = np.min(neighborTFBypass[:,0])
            self.numNeighborTFBypass = neighborTFBypass.shape[0]

            if not len(neighborTFBypass_2)==0:
                self.exMeanNeighborTFBypass_2 = np.average(neighborTFBypass_2[:,0])
                self.exMaxNeighborTFBypass_2 = np.max(neighborTFBypass_2[:,0])
                self.exMinNeighborTFBypass_2 = np.min(neighborTFBypass_2[:,0])
                self.numNeighborTFBypass_2 = neighborTFBypass_2.shape[0]
            if not len(neighborTFBypass_3)==0:
                self.exMeanNeighborTFBypass_3 = np.average(neighborTFBypass_3[:,0])
                self.exMaxNeighborTFBypass_3 = np.max(neighborTFBypass_3[:,0])
                self.exMinNeighborTFBypass_3 = np.min(neighborTFBypass_3[:,0])
                self.numNeighborTFBypass_3 = neighborTFBypass_3.shape[0]
            if not len(neighborTFBypass_H)==0:
                self.exMeanNeighborTFBypass_H = np.average(neighborTFBypass_H[:,0])
                self.exMaxNeighborTFBypass_H = np.max(neighborTFBypass_H[:,0])
                self.exMinNeighborTFBypass_H = np.min(neighborTFBypass_H[:,0])
                self.numNeighborTFBypass_H = neighborTFBypass_H.shape[0]

        if not len(neighborTargetBypass)==0:
            self.exMeanNeighborTargetBypass = np.average(neighborTargetBypass[:,0])
            self.exMaxNeighborTargetBypass = np.max(neighborTargetBypass[:,0])
            self.exMinNeighborTargetBypass = np.min(neighborTargetBypass[:,0])
            self.numNeighborTargetBypass = neighborTargetBypass.shape[0]

            if not len(neighborTargetBypass_2)==0:
                self.exMeanNeighborTargetBypass_2 = np.average(neighborTargetBypass_2[:,0])
                self.exMaxNeighborTargetBypass_2 = np.max(neighborTargetBypass_2[:,0])
                self.exMinNeighborTargetBypass_2 = np.min(neighborTargetBypass_2[:,0])
                self.numNeighborTargetBypass_2 = neighborTargetBypass_2.shape[0]
            if not len(neighborTargetBypass_3)==0:
                self.exMeanNeighborTargetBypass_3 = np.average(neighborTargetBypass_3[:,0])
                self.exMaxNeighborTargetBypass_3 = np.max(neighborTargetBypass_3[:,0])
                self.exMinNeighborTargetBypass_3 = np.min(neighborTargetBypass_3[:,0])
                self.numNeighborTargetBypass_3 = neighborTargetBypass_3.shape[0]
            if not len(neighborTargetBypass_H)==0:
                self.exMeanNeighborTargetBypass_H = np.average(neighborTargetBypass_H[:,0])
                self.exMaxNeighborTargetBypass_H = np.max(neighborTargetBypass_H[:,0])
                self.exMinNeighborTargetBypass_H = np.min(neighborTargetBypass_H[:,0])
                self.numNeighborTargetBypass_H = neighborTargetBypass_H.shape[0]

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

#condition9_2: if neighbors have TF: if have bypass: exMeanNeighborTFBypass_2>end1/exMeanNeighborTFBypass_2<end1
def condition9_2(gi):
    #exMeanNeighborTFBypass_2>end1
    if gi.exMeanNeighborTFBypass_2 > gi.exEnd1 :
        result = True   
    #exMeanNeighborTFBypass_2<end1
    else:
        result = False
    return result

#condition9_3: if neighbors have TF: if have bypass: exMeanNeighborTFBypass_3>end1/exMeanNeighborTFBypass_3<end1
def condition9_3(gi):
    #exMeanNeighborTFBypass_3>end1
    if gi.exMeanNeighborTFBypass_3 > gi.exEnd1 :
        result = True   
    #exMeanNeighborTFBypass_3<end1
    else:
        result = False
    return result

#condition9_H: if neighbors have TF: if have bypass: exMeanNeighborTFBypass_H>end1/exMeanNeighborTFBypass_H<end1
def condition9_H(gi):
    #exMeanNeighborTFBypass_H>end1
    if gi.exMeanNeighborTFBypass_H > gi.exEnd1 :
        result = True   
    #exMeanNeighborTFBypass_H<end1
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

#condition10_2: if neighbors have TF: if have bypass: exMeanNeighborTFBypass_2>end2/exMeanNeighborTFBypass_2<end2
def condition10_2(gi):
    #exMeanNeighborTFBypass_2>end2
    if gi.exMeanNeighborTFBypass_2 > gi.exEnd2 :
        result = True   
    #exMeanNeighborTFBypass_2<end2
    else:
        result = False
    return result

#condition10_3: if neighbors have TF: if have bypass: exMeanNeighborTFBypass_3>end2/exMeanNeighborTFBypass_3<end2
def condition10_3(gi):
    #exMeanNeighborTFBypass_3>end2
    if gi.exMeanNeighborTFBypass_3 > gi.exEnd2 :
        result = True   
    #exMeanNeighborTFBypass_3<end2
    else:
        result = False
    return result

#condition10_H: if neighbors have TF: if have bypass: exMeanNeighborTFBypass_H>end2/exMeanNeighborTFBypass_H<end2
def condition10_H(gi):
    #exMeanNeighborTFBypass_H>end2
    if gi.exMeanNeighborTFBypass_H > gi.exEnd2 :
        result = True   
    #exMeanNeighborTFBypass_H<end2
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

#condition11_2: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass_2>end1/exMeanNeighborTargetBypass_2<end1
def condition11_2(gi):
    #exMeanNeighborTargetBypass_2>end1
    if gi.exMeanNeighborTargetBypass_2 > gi.exEnd1 :
        result = True   
    #exMeanNeighborTargetBypass_2<end1
    else:
        result = False
    return result

#condition11_3: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass_3>end1/exMeanNeighborTargetBypass_3<end1
def condition11_3(gi):
    #exMeanNeighborTargetBypass_3>end1
    if gi.exMeanNeighborTargetBypass_3 > gi.exEnd1 :
        result = True   
    #exMeanNeighborTargetBypass_3<end1
    else:
        result = False
    return result

#condition11_H: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass_H>end1/exMeanNeighborTargetBypass_H<end1
def condition11_H(gi):
    #exMeanNeighborTargetBypass_H>end1
    if gi.exMeanNeighborTargetBypass_H > gi.exEnd1 :
        result = True   
    #exMeanNeighborTargetBypass_H<end1
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

#condition12_2: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass_2>end2/exMeanNeighborTargetBypass_2<end2
def condition12_2(gi):
    #exMeanNeighborTargetBypass_2>end2
    if gi.exMeanNeighborTargetBypass_2 > gi.exEnd2 :
        result = True   
    #exMeanNeighborTargetBypass_2<end2
    else:
        result = False
    return result

#condition12_3: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass_3>end2/exMeanNeighborTargetBypass_3<end2
def condition12_3(gi):
    #exMeanNeighborTargetBypass_3>end2
    if gi.exMeanNeighborTargetBypass_3 > gi.exEnd2 :
        result = True   
    #exMeanNeighborTargetBypass_3<end2
    else:
        result = False
    return result

#condition12_H: if neighbors have Target: if have bypass: exMeanNeighborTargetBypass_H>end2/exMeanNeighborTargetBypass_H<end2
def condition12_H(gi):
    #exMeanNeighborTargetBypass_H>end2
    if gi.exMeanNeighborTargetBypass_H > gi.exEnd2 :
        result = True   
    #exMeanNeighborTargetBypass_H<end2
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

#Not Good enough
#condition17: expression neighborTF > neighborTarget
def condition17(gi):
    #exMeanNeighborTF>exMeanNeighborTarget
    if gi.exMeanNeighborTF > gi.exMeanNeighborTarget :
        result = True   
    #exMeanNeighborTF<exMeanNeighborTarget
    else:
        result = False
    return result

#condition18:
#exMeanNeighborTFBypass_2,3,H
def condition18_1(gi):
    if gi.exMeanNeighborTFBypass_2 > gi.exMeanNeighborTFBypass_3 :
        result = True
    else:
        result = False
    return result

def condition18_2(gi):
    if gi.exMeanNeighborTFBypass_2 > gi.exMeanNeighborTFBypass_H :
        result = True
    else:
        result = False
    return result

def condition18_3(gi):
    if gi.exMeanNeighborTFBypass_3 > gi.exMeanNeighborTFBypass_H :
        result = True
    else:
        result = False
    return result

#condition19:
#exMeanNeighborTargetBypass_2,3,H
def condition19_1(gi):
    if gi.exMeanNeighborTargetBypass_2 > gi.exMeanNeighborTargetBypass_3 :
        result = True
    else:
        result = False
    return result

def condition19_2(gi):
    if gi.exMeanNeighborTargetBypass_2 > gi.exMeanNeighborTargetBypass_H :
        result = True
    else:
        result = False
    return result

def condition19_3(gi):
    if gi.exMeanNeighborTargetBypass_3 > gi.exMeanNeighborTargetBypass_H :
        result = True
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

# if condition3(gi):
#     num_TG_n12 = num_TG_n12 +1
# else:
#     if condition4(gi):
#         num_TG_1n2 = num_TG_1n2 +1
#     else:
#         num_TG_12n = num_TG_12n +1
#var1=num_TG_n12,var2=num_TG_1n2,var3=num_TG_12n
def neighborCase(gi,func1=condition3,func2=condition4,var1=ph1,var2=ph2,var3=ph3):
    if func1(gi):
        var1 = var1 +1
    else:
        if func2(gi):
            var2 = var2 +1
        else:
            var3 = var3 +1
    return var1, var2, var3


posGiList = generate_information(graphs_labels=pos_graphs_labels, graphs_features=pos_graphs_features)
# negGiList = generate_information(graphs_labels=neg_graphs_labels, graphs_features=neg_graphs_features)

#TT: TF-TF, TG: TF-Target
#n12: neighbor, end1, end2
#t12: neighborTF, end1, end2
#g12: neighborTarget, end1, end2
#tB12: neighborTF Bypass, end1, end2
#tB2_12: neighborTF Bypass label 2, end1, end2
#tB3_12: neighborTF Bypass label 3, end1, end2
#tBH_12: neighborTF Bypass label H, end1, end2
#tN12: neighborTF Bypassno, end1, end2
#gB12: neighborTarget Bypass, end1, end2
#gB2_12: neighborTarget Bypass label 2, end1, end2
#gB3_12: neighborTarget Bypass label 3, end1, end2
#gBH_12: neighborTarget Bypass label H, end1, end2
#gN12: neighborTarget Bypassno, end1, end2
num_TT=0
num_TT_tg=0
num_TT_gt=0
num_TT_n12=0
num_TT_1n2=0
num_TT_12n=0
num_TT_t12=0
num_TT_1t2=0
num_TT_12t=0
num_TT_tB12=0
num_TT_1tB2=0
num_TT_12tB=0
num_TT_tB2_12=0
num_TT_1tB2_2=0
num_TT_12tB2_=0
num_TT_tB3_12=0
num_TT_1tB3_2=0
num_TT_12tB3_=0
num_TT_tBH_12=0
num_TT_1tBH_2=0
num_TT_12tBH_=0
num_TT_tN12=0
num_TT_1tN2=0
num_TT_12tN=0
num_TT_g12=0
num_TT_1g2=0
num_TT_12g=0
num_TT_gB12=0
num_TT_1gB2=0
num_TT_12gB=0
num_TT_gB2_12=0
num_TT_1gB2_2=0
num_TT_12gB2_=0
num_TT_gB3_12=0
num_TT_1gB3_2=0
num_TT_12gB3_=0
num_TT_gBH_12=0
num_TT_1gBH_2=0
num_TT_12gBH_=0
num_TT_gN12=0
num_TT_1gN2=0
num_TT_12gN=0

num_TG=0
num_TG_tg=0
num_TG_gt=0
num_TG_12=0
num_TG_21=0
num_TG_n12=0
num_TG_1n2=0
num_TG_12n=0
num_TG_n21=0
num_TG_2n1=0
num_TG_21n=0
num_TG_t12=0
num_TG_1t2=0
num_TG_12t=0
num_TG_t21=0
num_TG_2t1=0
num_TG_21t=0
num_TG_tB12=0
num_TG_1tB2=0
num_TG_12tB=0
num_TG_tB2_12=0
num_TG_1tB2_2=0
num_TG_12tB2_=0
num_TG_tB3_12=0
num_TG_1tB3_2=0
num_TG_12tB3_=0
num_TG_tBH_12=0
num_TG_1tBH_2=0
num_TG_12tBH_=0
num_TG_tB21=0
num_TG_2tB1=0
num_TG_21tB=0
num_TG_tB2_21=0
num_TG_2tB2_1=0
num_TG_21tB2_=0
num_TG_tB3_21=0
num_TG_2tB3_1=0
num_TG_21tB3_=0
num_TG_tBH_21=0
num_TG_2tBH_1=0
num_TG_21tBH_=0
num_TG_tN12=0
num_TG_1tN2=0
num_TG_12tN=0
num_TG_tN21=0
num_TG_2tN1=0
num_TG_21tN=0
num_TG_g12=0
num_TG_1g2=0
num_TG_12g=0
num_TG_g21=0
num_TG_2g1=0
num_TG_21g=0
num_TG_gB12=0
num_TG_1gB2=0
num_TG_12gB=0
num_TG_gB2_12=0
num_TG_1gB2_2=0
num_TG_12gB2_=0
num_TG_gB3_12=0
num_TG_1gB3_2=0
num_TG_12gB3_=0
num_TG_gBH_12=0
num_TG_1gBH_2=0
num_TG_12gBH_=0
num_TG_gB21=0
num_TG_2gB1=0
num_TG_21gB=0
num_TG_gB2_21=0
num_TG_2gB2_1=0
num_TG_21gB2_=0
num_TG_gB3_21=0
num_TG_2gB3_1=0
num_TG_21gB3_=0
num_TG_gBH_21=0
num_TG_2gBH_1=0
num_TG_21gBH_=0
num_TG_gN12=0
num_TG_1gN2=0
num_TG_12gN=0
num_TG_gN21=0
num_TG_2gN1=0
num_TG_21gN=0

for gi in posGiList:
# for gi in negGiList:
    if condition1(gi):
        num_TT = num_TT + 1
        if gi.numNeighbor>0 and len(gi.neighborTF)>0 and len(gi.neighborTarget)>0:
            if condition17(gi):
                num_TT_tg = num_TT_tg +1
            else:
                num_TT_gt = num_TT_gt +1
        if condition2(gi):
            if gi.numNeighbor>0:
                num_TT_n12,num_TT_1n2,num_TT_12n = neighborCase(gi,func1=condition3,func2=condition4,var1=num_TT_n12,var2=num_TT_1n2,var3=num_TT_12n)
                if not len(gi.neighborTF)==0:
                    num_TT_t12,num_TT_1t2,num_TT_12t = neighborCase(gi,func1=condition5,func2=condition6,var1=num_TT_t12,var2=num_TT_1t2,var3=num_TT_12t)
                    if not len(gi.neighborTFBypass)==0:
                        num_TT_tB12,num_TT_1tB2,num_TT_12tB = neighborCase(gi,func1=condition9,func2=condition10,var1=num_TT_tB12,var2=num_TT_1tB2,var3=num_TT_12tB)
                        if not len(neighborTFBypass_2)==0:
                            num_TT_tB2_12,num_TT_1tB2_2,num_TT_12tB2_ = neighborCase(gi,func1=condition9_2,func2=condition10_2,var1=num_TT_tB2_12,var2=num_TT_1tB2_2,var3=num_TT_12tB2_)
                        if not len(neighborTFBypass_3)==0:
                            num_TT_tB3_12,num_TT_1tB3_2,num_TT_12tB3_ = neighborCase(gi,func1=condition9_3,func2=condition10_3,var1=num_TT_tB3_12,var2=num_TT_1tB3_2,var3=num_TT_12tB3_)
                        if not len(neighborTFBypass_H)==0:
                            num_TT_tBH_12,num_TT_1tBH_2,num_TT_12tBH_ = neighborCase(gi,func1=condition9_H,func2=condition10_H,var1=num_TT_tBH_12,var2=num_TT_1tBH_2,var3=num_TT_12tBH_)
                    if not len(gi.neighborTFBypassNo)==0:
                        num_TT_tN12,num_TT_1tN2,num_TT_12tN = neighborCase(gi,func1=condition13,func2=condition14,var1=num_TT_tN12,var2=num_TT_1tN2,var3=num_TT_12tN)
                if not len(gi.neighborTarget)==0:
                    num_TT_g12,num_TT_1g2,num_TT_12g = neighborCase(gi,func1=condition7,func2=condition8,var1=num_TT_g12,var2=num_TT_1g2,var3=num_TT_12g)
                    if not len(gi.neighborTargetBypass)==0:
                        num_TT_gB12,num_TT_1gB2,num_TT_12gB = neighborCase(gi,func1=condition11,func2=condition12,var1=num_TT_gB12,var2=num_TT_1gB2,var3=num_TT_12gB)
                        if not len(neighborTargetBypass_2)==0:
                            num_TT_gB2_12,num_TT_1gB2_2,num_TT_12gB2_ = neighborCase(gi,func1=condition11_2,func2=condition12_2,var1=num_TT_gB2_12,var2=num_TT_1gB2_2,var3=num_TT_12gB2_)
                        if not len(neighborTargetBypass_3)==0:
                            num_TT_gB3_12,num_TT_1gB3_2,num_TT_12gB3_ = neighborCase(gi,func1=condition11_3,func2=condition12_3,var1=num_TT_gB3_12,var2=num_TT_1gB3_2,var3=num_TT_12gB3_)
                        if not len(neighborTargetBypass_H)==0:
                            num_TT_gBH_12,num_TT_1gBH_2,num_TT_12gBH_ = neighborCase(gi,func1=condition11_H,func2=condition12_H,var1=num_TT_gBH_12,var2=num_TT_1gBH_2,var3=num_TT_12gBH_)
                    if not len(gi.neighborTargetBypassNo)==0:
                        num_TT_gN12,num_TT_1gN2,num_TT_12gN = neighborCase(gi,func1=condition15,func2=condition16,var1=num_TT_gN12,var2=num_TT_1gN2,var3=num_TT_12gN)
        else:
            if gi.numNeighbor>0:
                num_TT_n12,num_TT_1n2,num_TT_12n = neighborCase(gi,func1=condition4,func2=condition3,var1=num_TT_n12,var2=num_TT_1n2,var3=num_TT_12n)
                if not len(gi.neighborTF)==0:
                    num_TT_t12,num_TT_1t2,num_TT_12t = neighborCase(gi,func1=condition6,func2=condition5,var1=num_TT_t12,var2=num_TT_1t2,var3=num_TT_12t)
                    if not len(gi.neighborTFBypass)==0:
                        num_TT_tB12,num_TT_1tB2,num_TT_12tB = neighborCase(gi,func1=condition10,func2=condition9,var1=num_TT_tB12,var2=num_TT_1tB2,var3=num_TT_12tB)
                        if not len(gi.neighborTFBypass_2)==0:
                            num_TT_tB2_12,num_TT_1tB2_2,num_TT_12tB2_ = neighborCase(gi,func1=condition10_2,func2=condition9_2,var1=num_TT_tB2_12,var2=num_TT_1tB2_2,var3=num_TT_12tB2_)
                        if not len(gi.neighborTFBypass_3)==0:
                            num_TT_tB3_12,num_TT_1tB3_2,num_TT_12tB3_ = neighborCase(gi,func1=condition10_3,func2=condition9_3,var1=num_TT_tB3_12,var2=num_TT_1tB3_2,var3=num_TT_12tB3_)
                        if not len(gi.neighborTFBypass_H)==0:
                            num_TT_tBH_12,num_TT_1tBH_2,num_TT_12tBH_ = neighborCase(gi,func1=condition10_H,func2=condition9_H,var1=num_TT_tBH_12,var2=num_TT_1tBH_2,var3=num_TT_12tBH_)
                    if not len(gi.neighborTFBypassNo)==0:
                        num_TT_tN12,num_TT_1tN2,num_TT_12tN = neighborCase(gi,func1=condition14,func2=condition13,var1=num_TT_tN12,var2=num_TT_1tN2,var3=num_TT_12tN)
                if not len(gi.neighborTarget)==0:
                    num_TT_g12,num_TT_1g2,num_TT_12g = neighborCase(gi,func1=condition8,func2=condition7,var1=num_TT_g12,var2=num_TT_1g2,var3=num_TT_12g)
                    if not len(gi.neighborTargetBypass)==0:
                        num_TT_gB12,num_TT_1gB2,num_TT_12gB = neighborCase(gi,func1=condition12,func2=condition11,var1=num_TT_gB12,var2=num_TT_1gB2,var3=num_TT_12gB)
                        if not len(gi.neighborTargetBypass_2)==0:
                            num_TT_gB2_12,num_TT_1gB2_2,num_TT_12gB2_ = neighborCase(gi,func1=condition12_2,func2=condition11_2,var1=num_TT_gB2_12,var2=num_TT_1gB2_2,var3=num_TT_12gB2_)
                        if not len(gi.neighborTargetBypass_3)==0:
                            num_TT_gB3_12,num_TT_1gB3_2,num_TT_12gB3_ = neighborCase(gi,func1=condition12_3,func2=condition11_3,var1=num_TT_gB3_12,var2=num_TT_1gB3_2,var3=num_TT_12gB3_)
                        if not len(gi.neighborTargetBypass_H)==0:
                            num_TT_gBH_12,num_TT_1gBH_2,num_TT_12gBH_ = neighborCase(gi,func1=condition12_H,func2=condition11_H,var1=num_TT_gBH_12,var2=num_TT_1gBH_2,var3=num_TT_12gBH_)
                    if not len(gi.neighborTargetBypassNo)==0:
                        num_TT_gN12,num_TT_1gN2,num_TT_12gN = neighborCase(gi,func1=condition16,func2=condition15,var1=num_TT_gN12,var2=num_TT_1gN2,var3=num_TT_12gN)
                
    else:
        num_TG = num_TG + 1
        if gi.numNeighbor>0 and len(gi.neighborTF)>0 and len(gi.neighborTarget)>0:
            if condition17(gi):
                num_TG_tg = num_TG_tg +1
            else:
                num_TG_gt = num_TG_gt +1
        if condition2(gi):
            num_TG_12 = num_TG_12 +1
            if gi.numNeighbor>0:
                num_TG_n12,num_TG_1n2,num_TG_12n = neighborCase(gi,func1=condition3,func2=condition4,var1=num_TG_n12,var2=num_TG_1n2,var3=num_TG_12n)
                if not len(gi.neighborTF)==0:
                    num_TG_t12,num_TG_1t2,num_TG_12t = neighborCase(gi,func1=condition5,func2=condition6,var1=num_TG_t12,var2=num_TG_1t2,var3=num_TG_12t)
                    if not len(gi.neighborTFBypass)==0:
                        num_TG_tB12,num_TG_1tB2,num_TG_12tB = neighborCase(gi,func1=condition9,func2=condition10,var1=num_TG_tB12,var2=num_TG_1tB2,var3=num_TG_12tB)
                        if not len(gi.neighborTFBypass_2)==0:
                            num_TG_tB2_12,num_TG_1tB2_2,num_TG_12tB2_ = neighborCase(gi,func1=condition9_2,func2=condition10_2,var1=num_TG_tB2_12,var2=num_TG_1tB2_2,var3=num_TG_12tB2_)
                        if not len(gi.neighborTFBypass_3)==0:
                            num_TG_tB3_12,num_TG_1tB3_2,num_TG_12tB3_ = neighborCase(gi,func1=condition9_3,func2=condition10_3,var1=num_TG_tB3_12,var2=num_TG_1tB3_2,var3=num_TG_12tB3_)
                        if not len(gi.neighborTFBypass_H)==0:
                            num_TG_tBH_12,num_TG_1tBH_2,num_TG_12tBH_ = neighborCase(gi,func1=condition9_H,func2=condition10_H,var1=num_TG_tBH_12,var2=num_TG_1tBH_2,var3=num_TG_12tBH_)
                    if not len(gi.neighborTFBypassNo)==0:
                        num_TG_tN12,num_TG_1tN2,num_TG_12tN = neighborCase(gi,func1=condition13,func2=condition14,var1=num_TG_tN12,var2=num_TG_1tN2,var3=num_TG_12tN)                    
                if not len(gi.neighborTarget)==0:
                    num_TG_g12,num_TG_1g2,num_TG_12g = neighborCase(gi,func1=condition7,func2=condition8,var1=num_TG_g12,var2=num_TG_1g2,var3=num_TG_12g)
                    if not len(gi.neighborTargetBypass)==0:
                        num_TG_gB12,num_TG_1gB2,num_TG_12gB = neighborCase(gi,func1=condition11,func2=condition12,var1=num_TG_gB12,var2=num_TG_1gB2,var3=num_TG_12gB)
                        if not len(gi.neighborTargetBypass_2)==0:
                            num_TG_gB2_12,num_TG_1gB2_2,num_TG_12gB2_ = neighborCase(gi,func1=condition11_2,func2=condition12_2,var1=num_TG_gB2_12,var2=num_TG_1gB2_2,var3=num_TG_12gB2_)
                        if not len(gi.neighborTargetBypass_3)==0:
                            num_TG_gB3_12,num_TG_1gB3_2,num_TG_12gB3_ = neighborCase(gi,func1=condition11_3,func2=condition12_3,var1=num_TG_gB3_12,var2=num_TG_1gB3_2,var3=num_TG_12gB3_)
                        if not len(gi.neighborTargetBypass_H)==0:
                            num_TG_gBH_12,num_TG_1gBH_2,num_TG_12gBH_ = neighborCase(gi,func1=condition11_H,func2=condition12_H,var1=num_TG_gBH_12,var2=num_TG_1gBH_2,var3=num_TG_12gBH_)
                    if not len(gi.neighborTargetBypassNo)==0:
                        num_TG_gN12,num_TG_1gN2,num_TG_12gN = neighborCase(gi,func1=condition15,func2=condition16,var1=num_TG_gN12,var2=num_TG_1gN2,var3=num_TG_12gN) 
        else:
            num_TG_21 = num_TG_21 +1
            if gi.numNeighbor>0:
                num_TG_n21,num_TG_2n1,num_TG_21n = neighborCase(gi,func1=condition4,func2=condition3,var1=num_TG_n21,var2=num_TG_2n1,var3=num_TG_21n)
                if not len(gi.neighborTF)==0:
                    num_TG_t21,num_TG_2t1,num_TG_21t = neighborCase(gi,func1=condition6,func2=condition5,var1=num_TG_t21,var2=num_TG_2t1,var3=num_TG_21t)
                    if not len(gi.neighborTFBypass)==0:
                        num_TG_tB21,num_TG_2tB1,num_TG_21tB = neighborCase(gi,func1=condition10,func2=condition9,var1=num_TG_tB21,var2=num_TG_2tB1,var3=num_TG_21tB)
                        if not len(gi.neighborTFBypass_2)==0:
                            num_TG_tB2_21,num_TG_2tB2_1,num_TG_21tB2_ = neighborCase(gi,func1=condition10_2,func2=condition9_2,var1=num_TG_tB2_21,var2=num_TG_2tB2_1,var3=num_TG_21tB2_)
                        if not len(gi.neighborTFBypass_3)==0:
                            num_TG_tB3_21,num_TG_2tB3_1,num_TG_21tB3_ = neighborCase(gi,func1=condition10_3,func2=condition9_3,var1=num_TG_tB3_21,var2=num_TG_2tB3_1,var3=num_TG_21tB3_)
                        if not len(gi.neighborTFBypass_H)==0:
                            num_TG_tBH_21,num_TG_2tBH_1,num_TG_21tBH_ = neighborCase(gi,func1=condition10_H,func2=condition9_H,var1=num_TG_tBH_21,var2=num_TG_2tBH_1,var3=num_TG_21tBH_)
                    if not len(gi.neighborTFBypassNo)==0:
                        num_TG_tN21,num_TG_2tN1,num_TG_21tN = neighborCase(gi,func1=condition14,func2=condition13,var1=num_TG_tN21,var2=num_TG_2tN1,var3=num_TG_21tN)
                if not len(gi.neighborTarget)==0:
                    num_TG_g21,num_TG_2g1,num_TG_21g = neighborCase(gi,func1=condition8,func2=condition7,var1=num_TG_g21,var2=num_TG_2g1,var3=num_TG_21g)
                    if not len(gi.neighborTargetBypass)==0:
                        num_TG_gB21,num_TG_2gB1,num_TG_21gB = neighborCase(gi,func1=condition12,func2=condition11,var1=num_TG_gB21,var2=num_TG_2gB1,var3=num_TG_21gB)
                        if not len(gi.neighborTargetBypass_2)==0:
                            num_TG_gB2_21,num_TG_2gB2_1,num_TG_21gB2_ = neighborCase(gi,func1=condition12_2,func2=condition11_2,var1=num_TG_gB2_21,var2=num_TG_2gB2_1,var3=num_TG_21gB2_)
                        if not len(gi.neighborTargetBypass_3)==0:
                            num_TG_gB3_21,num_TG_2gB3_1,num_TG_21gB3_ = neighborCase(gi,func1=condition12_3,func2=condition11_3,var1=num_TG_gB3_21,var2=num_TG_2gB3_1,var3=num_TG_21gB3_)
                        if not len(gi.neighborTargetBypass_H)==0:
                            num_TG_gBH_21,num_TG_2gBH_1,num_TG_21gBH_ = neighborCase(gi,func1=condition12_H,func2=condition11_H,var1=num_TG_gBH_21,var2=num_TG_2gBH_1,var3=num_TG_21gBH_)
                    if not len(gi.neighborTargetBypassNo)==0:
                        num_TG_gN21,num_TG_2gN1,num_TG_21gN = neighborCase(gi,func1=condition16,func2=condition15,var1=num_TG_gN21,var2=num_TG_2gN1,var3=num_TG_21gN)
                

#TT: TF-TF, TG: TF-Target
#n12: neighbor, end1, end2
#t12: neighborTF, end1, end2
#g12: neighborTarget, end1, end2
#tB12: neighborTF Bypass, end1, end2
#tN12: neighborTF Bypassno, end1, end2
#gB12: neighborTarget Bypass, end1, end2
#gN12: neighborTarget Bypassno, end1, end2
print(num_TT)
print(num_TT_tg)
print(num_TT_gt)
print(num_TT_n12)
print(num_TT_1n2)
print(num_TT_12n)
print(num_TT_t12)
print(num_TT_1t2)
print(num_TT_12t)
print(num_TT_tB12)
print(num_TT_1tB2)
print(num_TT_12tB)
print(num_TT_tB2_12)
print(num_TT_1tB2_2)
print(num_TT_12tB2_)
print(num_TT_tB3_12)
print(num_TT_1tB3_2)
print(num_TT_12tB3_)
print(num_TT_tBH_12)
print(num_TT_1tBH_2)
print(num_TT_12tBH_)
print(num_TT_tN12)
print(num_TT_1tN2)
print(num_TT_12tN)
print(num_TT_g12)
print(num_TT_1g2)
print(num_TT_12g)
print(num_TT_gB12)
print(num_TT_1gB2)
print(num_TT_12gB)
print(num_TT_gB2_12)
print(num_TT_1gB2_2)
print(num_TT_12gB2_)
print(num_TT_gB3_12)
print(num_TT_1gB3_2)
print(num_TT_12gB3_)
print(num_TT_gBH_12)
print(num_TT_1gBH_2)
print(num_TT_12gBH_)
print(num_TT_gN12)
print(num_TT_1gN2)
print(num_TT_12gN)

print("****")
print(num_TG)
print(num_TG_tg)
print(num_TG_gt)
print(num_TG_12)
print(num_TG_21)
print(num_TG_n12)
print(num_TG_1n2)
print(num_TG_12n)
print(num_TG_n21)
print(num_TG_2n1)
print(num_TG_21n)
print(num_TG_t12)
print(num_TG_1t2)
print(num_TG_12t)
print(num_TG_t21)
print(num_TG_2t1)
print(num_TG_21t)
print(num_TG_tB12)
print(num_TG_1tB2)
print(num_TG_12tB)
print(num_TG_tB2_12)
print(num_TG_1tB2_2)
print(num_TG_12tB2_)
print(num_TG_tB3_12)
print(num_TG_1tB3_2)
print(num_TG_12tB3_)
print(num_TG_tBH_12)
print(num_TG_1tBH_2)
print(num_TG_12tBH_)
print(num_TG_tB21)
print(num_TG_2tB1)
print(num_TG_21tB)
print(num_TG_tB2_21)
print(num_TG_2tB2_1)
print(num_TG_21tB2_)
print(num_TG_tB3_21)
print(num_TG_2tB3_1)
print(num_TG_21tB3_)
print(num_TG_tBH_21)
print(num_TG_2tBH_1)
print(num_TG_21tBH_)
print(num_TG_tN12)
print(num_TG_1tN2)
print(num_TG_12tN)
print(num_TG_tN21)
print(num_TG_2tN1)
print(num_TG_21tN)
print(num_TG_g12)
print(num_TG_1g2)
print(num_TG_12g)
print(num_TG_g21)
print(num_TG_2g1)
print(num_TG_21g)
print(num_TG_gB12)
print(num_TG_1gB2)
print(num_TG_12gB)
print(num_TG_gB2_12)
print(num_TG_1gB2_2)
print(num_TG_12gB2_)
print(num_TG_gB3_12)
print(num_TG_1gB3_2)
print(num_TG_12gB3_)
print(num_TG_gBH_12)
print(num_TG_1gBH_2)
print(num_TG_12gBH_)
print(num_TG_gB21)
print(num_TG_2gB1)
print(num_TG_21gB)
print(num_TG_gB2_21)
print(num_TG_2gB2_1)
print(num_TG_21gB2_)
print(num_TG_gB3_21)
print(num_TG_2gB3_1)
print(num_TG_21gB3_)
print(num_TG_gBH_21)
print(num_TG_2gBH_1)
print(num_TG_21gBH_)
print(num_TG_gN12)
print(num_TG_1gN2)
print(num_TG_12gN)
print(num_TG_gN21)
print(num_TG_2gN1)
print(num_TG_21gN)





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