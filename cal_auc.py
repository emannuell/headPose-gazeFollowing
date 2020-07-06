from scipy.io import loadmat
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
import pandas as pd

# test_mat_file = '../GazeFollowData/test2_annotations.mat'
prediction_file = 'multi_scale_concat_heatmaps.npz'

fileName = '../../dataset/data_new/test_dataset.txt'
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
    error_list.append(score)
    gt_list.append(gt_heatmap)
    pred_list.append(pred)

print("mean", np.mean(error_list))
gt_list = np.stack(gt_list).reshape([-1])
pred_list = np.stack(pred_list).reshape([-1])

print("auc score")
score = roc_auc_score(gt_list, pred_list)
print(score)
