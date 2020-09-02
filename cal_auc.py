from scipy.io import loadmat
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

# test_mat_file = '../GazeFollowData/test2_annotations.mat'
# fileName = '../../dataset/data_new/test_dataset.txt'

prediction_file = '/home/emannuell/Documentos/mestrado/GazeFollowing/noHeadCode/multi_scale_concat_heatmaps.npz'
fileName = '/home/emannuell/Documentos/mestrado/dataset/data_new/test_dataset.txt'

df = pd.read_csv(fileName, header=None, sep=",", names=['image', 'imageCount', 'bbox0', 'bbox1', 'bbox2', 'bbox3', 'x_min', 'y_min', 'x_max', 'y_max', 'source', 'sourceName', 'y', 'p', 'r'])

# anns = loadmat(test_mat_file)
# gazes = anns['test_gaze']
x_max = df.x_max.to_numpy().reshape((-1, 1))
y_max = df.y_max.to_numpy().reshape((-1, 1))
gazes = np.hstack([x_max, y_max]).reshape(1, -1, 1, 2)
# eyes = anns['test_eyes']
x_min = df.x_min.to_numpy().reshape((-1, 1))
y_min = df.y_min.to_numpy().reshape((-1, 1))
eyes = np.hstack([x_min, y_min]).reshape(1, -1, 1, 2)
# N = anns['test_path'].shape[0]
N = df.image.to_numpy().reshape((-1, 1, 1)).shape[0]
print(N)

prediction = np.load(prediction_file)['heatmaps']
print(prediction.shape)

gt_list, pred_list = [], []
error_list = []
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(N):
    pred = prediction[i, :, :]
    eye_point = eyes[0, i][0]
    gt_points = gazes[0, i]
    pred = cv2.resize(pred, (5, 5))
    #pred[...] = 0.0
    #pred[2, 2] = 1.0
    gt_heatmap = np.zeros((5, 5))
    for gt_point in gt_points:
        x, y = list(map(int, list(gt_point * 5)))
        gt_heatmap[y, x] = 1.0

    score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), pred.reshape([-1]))
    # print('Score: ', score)
    error_list.append(score)
    # print('gt Heatmap: ', gt_heatmap)
    gt_list.append(gt_heatmap)
    # print('Pred: ', pred)
    pred_list.append(pred)
    # print('=====')
    # Compute ROC curve and ROC area
    fpr[i], tpr[i], _ = roc_curve(gt_heatmap.reshape([-1]).astype(np.int32), pred.reshape([-1]).astype(np.int32))
    roc_auc[i] = auc(fpr[i], tpr[i])


print("mean", np.mean(error_list))
gt_list = np.stack(gt_list).reshape([-1])
pred_list = np.stack(pred_list).reshape([-1])

print("auc score")
score = roc_auc_score(gt_list, pred_list)
print(score)

# Compute micro-average ROC curve and ROC area
fpr, tpr, threshold = roc_curve(gt_list, pred_list)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
