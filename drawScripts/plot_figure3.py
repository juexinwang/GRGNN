import matplotlib.pyplot as plt
import numpy as np

x=[0,1,2,3,4,5,6,7,8,9]
acc = [0.704, 0.71125, 0.64775, 0.693, 0.7315, 0.702, 0.69775, 0.70375, 0.72275, 0.7095]
precision = [0.757901391, 0.748091603, 0.8113804, 0.72183908, 0.811994609 ,0.764052288, 0.721568627, 0.795076032, 0.819813352, 0.805839416]
recall = [0.5995, 0.637, 0.385, 0.628, 0.6025, 0.5845, 0.644, 0.549, 0.571, 0.552]
mcc = [0.417213902, 0.42723703, 0.347322782, 0.389303633, 0.479224174, 0.41563984, 0.397805248, 0.428541633, 0.467553842, 0.441474741]
plt.plot([1, 1], [0, 1.0], linestyle='--')
plt.plot(x, acc, 'c', label='Accuracy')
plt.plot(x, precision, ':.g', label='Precision')
plt.plot(x, recall, ':k', label='Recall')
plt.plot(x, mcc, '-.r', label='MCC')
plt.xlabel('Number of Hops')
plt.ylim([0.0,1.0])
plt.xticks(np.arange(0, 10, step=1))
plt.title('GRGNN performances in different hops')
plt.legend(loc="lower right")
plt.savefig('Figure3.jpeg', dpi=300)