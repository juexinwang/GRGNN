import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

#43
test_size=2000
test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/single_neg_43_.npy')
test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/single_prob_43_.npy')
# test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/enresult_43.npy')
# test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/logits_43.npy')
# test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/svm_true_y_43_all.npy')
# test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/svm_y_score_43_all.npy')
# test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/rf_true_y_43_all.npy')
# test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/rf_y_score_43_all.npy')

#34
# test_size=3795
# test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/single_neg_34.npy')
# test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/single_prob_34.npy')
# test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/enresult_34.npy')
# test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/logits_34.npy')
# test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/svm_true_y_34_all.npy')
# test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/svm_y_score_34_all.npy')
# test_neg_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/rf_true_y_34_all.npy')
# test_prob_agent0=np.load('/home/wangjue/geneNetwork/SEAL/Python/nnpy/rf_y_score_34_all.npy')

#GRGNN
outputprob=abs(test_prob_agent0)
sort_index = np.argsort(outputprob)
use_index=sort_index[-test_size:]
dict_agent0={}
for i in test_neg_agent0:
	dict_agent0[i]=0
	
count = 0
for i in use_index:
	if i not in dict_agent0:
		count = count+1
		
print(count)

#method2
# outputprob=test_prob_agent0
# sort_index = np.argsort(outputprob)
# use_index=sort_index[:test_size]
	
# count = 0
# for i in np.arange(test_size):
# 	if use_index[i]<test_size:
# 		count=count+1
		
# print(count)


#SVM
# outputprob=test_prob_agent0
# sort_index = np.argsort(outputprob)
# use_index=sort_index[-test_size:]
	
# count = 0
# for i in np.arange(test_size):
# 	if use_index[i]<test_size:
# 		count=count+1
			
# print(count)ppy


#RF 
# outputprob=[]
# for i in np.arange(len(test_prob_agent0)):
# 	outputprob.append(test_prob_agent0[i][0]-test_prob_agent0[i][1])
# sort_index = np.argsort(outputprob)
# use_index=sort_index[:test_size]
	
# count = 0
# for i in np.arange(test_size):
# 	if use_index[i]<test_size:
# 		count=count+1
			
# print(count)
