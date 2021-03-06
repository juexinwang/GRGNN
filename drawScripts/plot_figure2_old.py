import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from inspect import signature

true_y_svm =np.load('../nnpy/svm_true_y_34.npy')
y_score_svm=np.load('../nnpy/svm_y_score_34.npy')
precision_svm, recall_svm, _ = precision_recall_curve(true_y_svm, y_score_svm)
auc_svm = roc_auc_score(true_y_svm, y_score_svm)
print('AUC: %.3f' % auc_svm)
fpr_svm, tpr_svm, _ = roc_curve(true_y_svm, y_score_svm)

true_y_svm43 =np.load('../nnpy/svm_true_y_43.npy')
y_score_svm43=np.load('../nnpy/svm_y_score_43.npy')
precision_svm43, recall_svm43, _ = precision_recall_curve(true_y_svm43, y_score_svm43)
auc_svm43 = roc_auc_score(true_y_svm43, y_score_svm43)
print('AUC: %.3f' % auc_svm43)
fpr_svm43, tpr_svm43, _ = roc_curve(true_y_svm43, y_score_svm43)

true_y_genie =np.load('../nnpy/genie3_true_y_4.npy')
y_score_genie =np.load('../nnpy/genie3_y_score_4.npy')
precision_genie, recall_genie, _ = precision_recall_curve(true_y_genie, y_score_genie)
auc_genie = roc_auc_score(true_y_genie, y_score_genie)
print('AUC: %.3f' % auc_genie)
fpr_genie, tpr_genie, _ = roc_curve(true_y_genie, y_score_genie)

true_y_genie3 =np.load('../nnpy/genie3_true_y_3.npy')
y_score_genie3 =np.load('../nnpy/genie3_y_score_3.npy')
precision_genie3, recall_genie3, _ = precision_recall_curve(true_y_genie3, y_score_genie3)
auc_genie3 = roc_auc_score(true_y_genie3, y_score_genie3)
print('AUC: %.3f' % auc_genie3)
fpr_genie3, tpr_genie3, _ = roc_curve(true_y_genie3, y_score_genie3)

true_y_gnn =np.load('../nnpy/GNNPC_true_y_34.npy')
y_score_gnn =np.load('../nnpy/GNNPC_y_score_34.npy')
precision_gnn, recall_gnn, _ = precision_recall_curve(true_y_gnn, y_score_gnn)
auc_gnn = roc_auc_score(true_y_gnn, y_score_gnn)
print('AUC: %.3f' % auc_gnn)
fpr_gnn, tpr_gnn, _ = roc_curve(true_y_gnn, y_score_gnn)

true_y_gnn3 =np.load('../nnpy/GNNPC_true_y_43.npy')
y_score_gnn3 =np.load('../nnpy/GNNPC_y_score_43.npy')
precision_gnn3, recall_gnn3, _ = precision_recall_curve(true_y_gnn3, y_score_gnn3)
auc_gnn3 = roc_auc_score(true_y_gnn3, y_score_gnn3)
print('AUC: %.3f' % auc_gnn3)
fpr_gnn3, tpr_gnn3, _ = roc_curve(true_y_gnn3, y_score_gnn3)


true_y_rf =np.load('../nnpy/rf_true_y_34_all.npy')
true_y_rf = true_y_rf[:7590]
tmp =np.load('../nnpy/rf_y_score_34_all.npy')
y_score_rf =[]
for i in np.arange(7590):
    y_score_rf.append(tmp[i][0]-tmp[i][1])
precision_rf, recall_rf, _ = precision_recall_curve(true_y_rf, y_score_rf)
auc_rf = roc_auc_score(true_y_rf, y_score_rf)
print('AUC: %.3f' % auc_rf)
fpr_rf, tpr_rf, _ = roc_curve(true_y_rf, y_score_rf)

true_y_rf3 =np.load('../nnpy/rf_true_y_43_all.npy')
true_y_rf3 = true_y_rf3[:4000]
tmp =np.load('../nnpy/rf_y_score_43_all.npy')
y_score_rf3 =[]
for i in np.arange(4000):
    y_score_rf3.append(tmp[i][0]-tmp[i][1])
precision_rf3, recall_rf3, _ = precision_recall_curve(true_y_rf, y_score_rf)
auc_rf3 = roc_auc_score(true_y_rf3, y_score_rf3)
print('AUC: %.3f' % auc_rf3)
fpr_rf3, tpr_rf3, _ = roc_curve(true_y_rf3, y_score_rf3)


fig, axs = plt.subplots(2, 2)


axs[0, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
axs[0, 0].plot(fpr_svm43, tpr_svm43, '-.b', label='SVM')
# axs[0, 0].plot(fpr_rf3, tpr_rf3, 'k', label='RF')
axs[0, 0].plot(fpr_genie3, tpr_genie3, ':g', label='GENIE3')
axs[0, 0].plot(fpr_gnn3, tpr_gnn3, 'r', label='GRGNN')
axs[0, 0].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
axs[0, 0].set_title('E.coli')
axs[0, 0].legend()


axs[0, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axs[0, 1].plot(fpr_svm, tpr_svm, '-.b', label='SVM')
# axs[0, 1].plot(fpr_rf, tpr_rf, 'k', label='RF')
axs[0, 1].plot(fpr_genie, tpr_genie, ':g', label='GENIE3')
axs[0, 1].plot(fpr_gnn, tpr_gnn, 'r', label='GRGNN')
axs[0, 1].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
axs[0, 1].set_title('S.cerevisiae')
axs[0, 1].legend()


axs[1, 0].plot([0, 1], [0.5, 0.5], linestyle='--')
axs[1, 0].plot(recall_svm43, precision_svm43, '-.b', label='SVM')
# axs[1, 0].plot(recall_rf3, precision_rf3, 'k', label='RF')
axs[1, 0].plot(recall_genie3, precision_genie3, ':g', label='GENIE3')
axs[1, 0].plot(recall_gnn3, precision_gnn3, 'r', label='GRGNN')
axs[1, 0].set(xlabel='Recall', ylabel='Precision')
axs[1, 0].set_title('E.coli')
axs[1, 0].legend(loc="lower right")

axs[1, 1].plot([0, 1], [0.5, 0.5], linestyle='--')
axs[1, 1].plot(recall_svm, precision_svm, '-.b', label='SVM')
# axs[1, 1].plot(recall_rf, precision_rf, 'k', label='RF')
axs[1, 1].plot(recall_genie, precision_genie, ':g', label='GENIE3')
axs[1, 1].plot(recall_gnn, precision_gnn, 'r', label='GRGNN')
axs[1, 1].set(xlabel='Recall', ylabel='Precision')
axs[1, 1].set_title('S.cerevisiae')
axs[1, 1].legend(loc="lower right")

for ax in axs.flat:
    ax.label_outer()

plt.savefig('Figure2.jpeg', dpi=300)



# fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])