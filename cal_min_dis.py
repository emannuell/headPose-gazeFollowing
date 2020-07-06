from scipy.io import loadmat
import numpy as np
import pandas as pd
# test_mat_file = '../GazeFollowData/test2_annotations.mat'
prediction_file = 'multi_scale_concat_prediction.npz'

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

prediction = np.load(prediction_file)['info_list']
print(prediction.shape)

error_list = []
for i in range(N):
    pred = prediction[i, :]
    eye_point = eyes[0, i][0]
    gt_points = gazes[0, i] 
    dis_list = []
    for gt_point in gt_points:
        #print(pred, gt_point, eye_point)
        f_error = pred - gt_point
        f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

        # angle 
        f_direction = pred - eye_point
        gt_direction = gt_point - eye_point

        norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
        norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5

        f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                    (norm_gt * norm_f + 1e-6)
        f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
        f_angle = np.arccos(f_cos_sim) * 180 / np.pi
        dis_list.append([f_dist, f_angle])
    error_list.append(np.min(np.array(dis_list), axis=0))

print("Min Dist and Min Angle:")
print(np.array(error_list).mean(axis=0))
