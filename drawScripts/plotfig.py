import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from inspect import signature

true_y_svm =np.load('svm_true_y_34.npy')
y_score_svm=np.load('svm_y_score_34.npy')
precision_svm, recall_svm, _ = precision_recall_curve(true_y_svm, y_score_svm)
auc_svm = roc_auc_score(true_y_svm, y_score_svm)
print('AUC: %.3f' % auc_svm)
fpr_svm, tpr_svm, _ = roc_curve(true_y_svm, y_score_svm)

true_y_svm43 =np.load('svm_true_y_43.npy')
y_score_svm43=np.load('svm_y_score_43.npy')
precision_svm43, recall_svm43, _ = precision_recall_curve(true_y_svm43, y_score_svm43)
auc_svm43 = roc_auc_score(true_y_svm43, y_score_svm43)
print('AUC: %.3f' % auc_svm43)
fpr_svm43, tpr_svm43, _ = roc_curve(true_y_svm43, y_score_svm43)

true_y_genie =np.load('/home/wangjue/Biclustering/genenetrfn/genie3_true_y_4.npy')
y_score_genie =np.load('/home/wangjue/Biclustering/genenetrfn/genie3_y_score_4.npy')
precision_genie, recall_genie, _ = precision_recall_curve(true_y_genie, y_score_genie)
auc_genie = roc_auc_score(true_y_genie, y_score_genie)
print('AUC: %.3f' % auc_genie)
fpr_genie, tpr_genie, _ = roc_curve(true_y_genie, y_score_genie)

true_y_genie3 =np.load('/home/wangjue/Biclustering/genenetrfn/genie3_true_y_3.npy')
y_score_genie3 =np.load('/home/wangjue/Biclustering/genenetrfn/genie3_y_score_3.npy')
precision_genie3, recall_genie3, _ = precision_recall_curve(true_y_genie3, y_score_genie3)
auc_genie3 = roc_auc_score(true_y_genie3, y_score_genie3)
print('AUC: %.3f' % auc_genie3)
fpr_genie3, tpr_genie3, _ = roc_curve(true_y_genie3, y_score_genie3)

true_y_gnn =np.load('GNNPC_true_y_34.npy')
y_score_gnn =np.load('GNNPC_y_score_34.npy')
precision_gnn, recall_gnn, _ = precision_recall_curve(true_y_gnn, y_score_gnn)
auc_gnn = roc_auc_score(true_y_gnn, y_score_gnn)
print('AUC: %.3f' % auc_gnn)
fpr_gnn, tpr_gnn, _ = roc_curve(true_y_gnn, y_score_gnn)

true_y_gnn3 =np.load('GNNPC_true_y_43.npy')
y_score_gnn3 =np.load('GNNPC_y_score_43.npy')
precision_gnn3, recall_gnn3, _ = precision_recall_curve(true_y_gnn3, y_score_gnn3)
auc_gnn3 = roc_auc_score(true_y_gnn3, y_score_gnn3)
print('AUC: %.3f' % auc_gnn3)
fpr_gnn3, tpr_gnn3, _ = roc_curve(true_y_gnn3, y_score_gnn3)

plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall_svm, precision_svm, 'b', label='SVM: S.cerevisae')
plt.plot(recall_genie, precision_genie, 'g', label='genie3: S.cerevisae')
plt.plot(recall_gnn, precision_gnn, 'r', label='GRGNN: S.cerevisae')
plt.plot(recall_svm43, precision_svm43, 'c', label='SVM: E.coli')
plt.plot(recall_genie3, precision_genie3, 'm', label='genie3: E.coli')
plt.plot(recall_gnn3, precision_gnn3, 'k', label='GRGNN: E.coli')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
# plt.show()
plt.savefig('pr.png')
plt.clf()



plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plot the precision-recall curve for the model
plt.plot(fpr_svm, tpr_svm, 'b', label='SVM: S.cerevisae')
plt.plot(fpr_genie, tpr_genie, 'g', label='genie3: S.cerevisae')
plt.plot(fpr_gnn, tpr_gnn, 'r', label='GRGNN: S.cerevisae')
plt.plot(fpr_svm43, tpr_svm43, 'c', label='SVM: E.coli')
plt.plot(fpr_genie3, tpr_genie3, 'm', label='genie3: E.coli')
plt.plot(fpr_gnn3, tpr_gnn3, 'k', label='GRGNN: E.coli')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
# plt.show()
plt.savefig('roc.png')



# fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])