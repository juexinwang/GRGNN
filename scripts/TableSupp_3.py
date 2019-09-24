import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/single_neg_34.npy')
test_neg_agent1=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/single_neg_34_.npy')
test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/single_prob_34.npy')
test_prob_agent1=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/single_prob_34_.npy')

#34
testpos_size = 3795

#43
# testpos_size = 2000

total_size = len(test_prob_agent0)
testneg_size = total_size - testpos_size


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

enresult=[]

for i in np.arange(len(test_prob_agent0)):
    if i<testpos_size: #positive part
        if i in dic_agent0 or i in dic_agent1:
            if test_prob_agent0[i]*test_prob_agent1[i]>0:
                # both wrong
                bothwrong = bothwrong + 1
                enresult.append(i)
            else:
                if abs(test_prob_agent0[i])>abs(test_prob_agent1[i]):
                    if i in dic_agent0 and i not in dic_agent1:
                        uncorrected = uncorrected +1
                        enresult.append(i)
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
                        enresult.append(i)  
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
                enresult.append(i)
            else:
                if abs(test_prob_agent0[i])>abs(test_prob_agent1[i]):
                    if i in dic_agent0 and i not in dic_agent1:
                        uncorrected = uncorrected +1
                        enresult.append(i)
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
                        enresult.append(i)  
                        tn0 = tn0 + 1                  
        else:
            count = count +1
            tn = tn +1 
            tn0 = tn0 + 1
            tn1 = tn1 + 1

print(count)
print(bothwrong)
print(corrected)
print(uncorrected)
print(len(enresult))

allstr = str(float((tp+tn)/total_size))+"\t"+str(tp)+"\t"+str(testpos_size-tp)+"\t"+str(tn)+"\t"+str(testneg_size-tn)
agent0_str = str(float((tp0+tn0)/total_size))+"\t"+str(tp0)+"\t"+str(testpos_size-tp0)+"\t"+str(tn0)+"\t"+str(testneg_size-tn0)
agent1_str = str(float((tp1+tn1)/total_size))+"\t"+str(tp1)+"\t"+str(testpos_size-tp1)+"\t"+str(tn1)+"\t"+str(testneg_size-tn1)
result = str(float(count/total_size))
print(allstr+"\t"+agent0_str+"\t"+agent1_str)

logits=0.5*test_prob_agent0+0.5*test_prob_agent1
np.save('/home/wangjue/geneNetwork/SEAL/Python/nnpy/enresult_34.npy',enresult)
np.save('/home/wangjue/geneNetwork/SEAL/Python/nnpy/logits_34.npy',logits)

