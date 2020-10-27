import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

# Read Genie3 results
count = 0
genieDict={}
genieList=[]
#with open ('/home/wangjue/Biclustering/genenetrfn/genie3/1_ranking.txt') as f:
with open ('/home/wangjue/Biclustering/genenetrfn/genie3/3_ranking.txt') as f:
#with open ('/home/wangjue/Biclustering/genenetrfn/genie3/4_ranking.txt') as f:
    lines = f.readlines() 
    for line in lines:        
        words = line.split()
        node1 = int(words[0][1:])
        node2 = int(words[1][1:])
        if node1 < node2:
            genieDict[str(node1)+","+str(node2)]=float(words[2])
            genieList.append(str(node1)+","+str(node2))
        elif node1 > node2: 
            genieDict[str(node2)+","+str(node1)]=float(words[2])
            genieList.append(str(node2)+","+str(node1))            
f.close()


# Read Results
trueList=[]
trueDict={}
#with open('/home/wangjue/biodata/DREAM5_network_inference_challenge/Network1/gold_standard/Network1_GoldStandard.tsv') as f:
with open('/home/wangjue/biodata/DREAM5_network_inference_challenge/Network3/gold standard/Network3_GoldStandard.tsv') as f:
#with open('/home/wangjue/biodata/DREAM5_network_inference_challenge/Network4/gold standard/Network4_GoldStandard.tsv') as f:
    lines = f.readlines()
    for line in lines:
        words = line.split()
        node1 = int(words[0][1:])
        node2 = int(words[1][1:])
        if node1 < node2:
            trueList.append(str(node1)+","+str(node2))
            trueDict[str(node1)+","+str(node2)]=1
        elif node1 > node2:
            trueList.append(str(node2)+","+str(node1))
            trueDict[str(node2)+","+str(node1)]=1       
f.close()

outList=[]
test_y =[]
pred_score=[]
# compare result
for genie in genieList:
    pred_score.append(genieDict[genie])
    if genie in trueDict:
        test_y.append(1)
        outList.append(1)
    else:
        test_y.append(0)


print("Length of found genie3: "+str(len(trueList))+" "+str(len(outList))+" "+str(len(genieList))+" "+str(len(outList)/len(genieList)))

np.save('genie3_true_y_3.npy',test_y)
np.save('genie3_y_score_3.npy',pred_score)

#np.save('genie3_true_y_4.npy',test_y)
#np.save('genie3_y_score_4.npy',pred_score)

#precision, recall, _ = precision_recall_curve(test_y, pred_score)
## plot no skill
#plt.plot([0, 1], [0.5, 0.5], linestyle='--')
## plot the precision-recall curve for the model
#plt.plot(recall, precision, marker='.')
## show the plot
#plt.show()
#plt.savefig('test.png')