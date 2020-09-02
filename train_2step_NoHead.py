import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging
import pandas as pd

from scipy import signal
from sklearn import preprocessing

from utils import data_transforms
from utils import get_paste_kernel, kernel_map


# log setting
log_dir = 'output/noStep01/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = log_dir + 'noStep01.log'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


class GazeDataset(Dataset):
    def __init__(self, root_dir, training='train'):
        assert (training in set(['train', 'test']))
        self.root_dir = root_dir
        self.training = training

        if training == 'train':
            fileName = self.root_dir + 'train_dataset.txt'
        else: 
            fileName = self.root_dir + 'test_dataset.txt'
        df = pd.read_csv(fileName, header=None, sep=",", names=['image', 'imageCount', 'bbox0', 'bbox1', 'bbox2', 'bbox3', 'x_min', 'y_min', 'x_max', 'y_max', 'source', 'sourceName', 'y', 'p', 'r'])
        
        # head pose data normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        y = min_max_scaler.fit_transform(df.y.to_numpy().reshape((-1, 1)))
        p = min_max_scaler.fit_transform(df.p.to_numpy().reshape((-1, 1)))
        r = min_max_scaler.fit_transform(df.r.to_numpy().reshape((-1, 1)))
        self.headPose = np.hstack([y, p, r]).reshape(1, -1, 1, 3)
        
        # df.eye = pd.DataFrame(y_scaled)
        bbox0 = df.bbox0.to_numpy().reshape((-1, 1))
        bbox1 = df.bbox1.to_numpy().reshape((-1, 1))
        bbox2 = df.bbox2.to_numpy().reshape((-1, 1))
        bbox3 = df.bbox3.to_numpy().reshape((-1, 1))
        self.bboxes = np.hstack([bbox0, bbox1, bbox2, bbox3]).reshape((1, -1, 4))

        x_max = df.x_max.to_numpy().reshape((-1, 1))
        y_max = df.y_max.to_numpy().reshape((-1, 1))
        self.gazes = np.hstack([x_max, y_max]).reshape(1, -1, 1, 2)
        # print(self.gazes)

        self.paths = df.image.to_numpy().reshape((-1, 1, 1))
        self.image_num = self.paths.shape[0]
        # self.paths = [paths]).reshape(1, -1)
        # print(self.paths)

        x_min = df.x_min.to_numpy().reshape((-1, 1))
        y_min = df.y_min.to_numpy().reshape((-1, 1))
        self.eyes = np.hstack([x_min, y_min]).reshape(1, -1, 1, 2)

        logging.info('%s contains %d images' % (fileName, self.image_num))

    def generate_data_field(self, eye_point):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = eye_point
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image_path = self.paths[idx][0][0]
        image_path = os.path.join(self.root_dir, image_path)

        eye = self.eyes[0, idx][0]
        headPose = self.headPose[0, idx][0]

        # todo: process gaze differently for training or testing
        # print(self.gazes[0, idx])
        gaze = self.gazes[0, idx].mean(axis=0)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # process image for saliency net
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = data_transforms[self.training](image)

        # generate gaze field
        gaze_field = self.generate_data_field(eye_point=eye)
        # generate heatmap
        heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))
        '''
        direction = gaze - eye
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0

        direction = direction / norm
        '''
        sample = {'image' : image,
                  'eye_position': torch.FloatTensor(eye),
                  'gaze_field': torch.from_numpy(gaze_field),
                  'head_pose': torch.FloatTensor(headPose),
                  'gt_position': torch.FloatTensor(gaze),
                  'gt_heatmap': torch.FloatTensor(heatmap).unsqueeze(0)}

        return sample


cosine_similarity = nn.CosineSimilarity()
mse_distance = nn.MSELoss()
bce_loss = nn.BCELoss()

def F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap):
    # point loss
    heatmap_loss = bce_loss(predict_heatmap, gt_heatmap)

    # angle loss
    gt_direction = gt_position - eye_position
    middle_angle_loss = torch.mean(1 - cosine_similarity(direction, gt_direction))

    return heatmap_loss, middle_angle_loss


def test(net, test_data_loader):
    net.eval()
    total_loss = []
    total_error = []
    info_list = []
    heatmaps = []

    for data in test_data_loader:
        image, gaze_field, eye_position, gt_position, gt_heatmap, head_pose = \
            data['image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap'], data['head_pose']
        image, gaze_field, eye_position, gt_position, gt_heatmap, head_pose = \
            map(lambda x: Variable(x.cuda(), volatile=True), [image, gaze_field, eye_position, gt_position, gt_heatmap, head_pose])

        direction, predict_heatmap = net([image, gaze_field, eye_position, head_pose])

        heatmap_loss, m_angle_loss = F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

        loss = heatmap_loss + m_angle_loss


        total_loss.append([heatmap_loss.data, m_angle_loss.data, loss.data])
        # logging.info('loss: %.5lf, %.5lf, %.5lf'%(heatmap_loss.data, m_angle_loss.data, loss.data))

        middle_output = direction.cpu().data.numpy()
        final_output = predict_heatmap.cpu().data.numpy()
        target = gt_position.cpu().data.numpy()
        eye_position = eye_position.cpu().data.numpy()
        for m_direction, f_point, gt_point, eye_point in \
            zip(middle_output, final_output, target, eye_position):
            f_point = f_point.reshape([224 // 4, 224 // 4])
            heatmaps.append(f_point)

            h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
            f_point = np.array([w_index / 56., h_index / 56.])

            f_error = f_point - gt_point
            f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

            # angle
            f_direction = f_point - eye_point
            gt_direction = gt_point - eye_point

            norm_m = (m_direction[0] **2 + m_direction[1] ** 2 ) ** 0.5
            norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
            norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5
            
            m_cos_sim = (m_direction[0]*gt_direction[0] + m_direction[1]*gt_direction[1]) / \
                        (norm_gt * norm_m + 1e-6)
            m_cos_sim = np.maximum(np.minimum(m_cos_sim, 1.0), -1.0)
            m_angle = np.arccos(m_cos_sim) * 180 / np.pi

            f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                        (norm_gt * norm_f + 1e-6)
            f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            f_angle = np.arccos(f_cos_sim) * 180 / np.pi

            
            total_error.append([f_dist, m_angle, f_angle])
            info_list.append(list(f_point))
    info_list = np.array(info_list)
    #np.savez('multi_scale_concat_prediction.npz', info_list=info_list)

    #heatmaps = np.stack(heatmaps)
    #np.savez('multi_scale_concat_heatmaps.npz', heatmaps=heatmaps)

    logging.info('average loss : %s'%str(np.mean(np.array(total_loss), axis=0)))
    logging.info('average error: %s'%str(np.mean(np.array(total_error), axis=0)))

    net.train()
    return


def main():
    dataset_path = '/home/emannuell/Documentos/mestrado/dataset/data_new/'
    output_path = 'output/noStep01'
    train_set = GazeDataset(root_dir=dataset_path,
                            training='train')
    train_data_loader = DataLoader(train_set, batch_size=10,
                                   shuffle=True, num_workers=4)

    test_set = GazeDataset(root_dir=dataset_path,
                           training='test')
    test_data_loader = DataLoader(test_set, batch_size=2,
                                  shuffle=False, num_workers=4)

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    resume_training = False
    if resume_training:
        pretrained_dict = torch.load('model/pretrained_model.pkl')
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        test(net, test_data_loader)
        exit()

    method = 'Adam'
    learning_rate = 0.0001

    optimizer_s1 = optim.Adam([{'params': net.module.head_pose_transform.parameters(), 
                                'initial_lr': learning_rate},
                               {'params': net.module.eye_position_transform.parameters(), 
                                'initial_lr': learning_rate},
                               {'params': net.module.fusion.parameters(), 
                                'initial_lr': learning_rate},
                                {'params': net.module.fpn_net.parameters(), 
                                'initial_lr': learning_rate}],
                                lr=learning_rate, weight_decay=0.0001)
    # optimizer_s1 = optim.Adam([{'params': net.parameters(), 
    #                             'initial_lr': learning_rate}],
    #                             lr=learning_rate, weight_decay=0.0001)

    lr_scheduler_s1 = optim.lr_scheduler.StepLR(optimizer_s1, step_size=5, gamma=0.1, last_epoch=-1)


    max_epoch = 30

    epoch = 0

    while epoch < max_epoch:
        print('EPOCH => ', epoch)
        lr_scheduler = lr_scheduler_s1
        optimizer = optimizer_s1
        
        lr_scheduler.step()
        ep_m_angle_loss = []
        ep_heatmap_loss = []
        running_loss = []
        for i, data in tqdm(enumerate(train_data_loader)):
            image, gaze_field, eye_position, gt_position, gt_heatmap, head_pose = \
                data['image'],  data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap'], data['head_pose']
            image, gaze_field, eye_position, gt_position, gt_heatmap, head_pose = \
                map(lambda x: Variable(x.cuda()), [image, gaze_field, eye_position, gt_position, gt_heatmap, head_pose])
            
            optimizer.zero_grad()
            direction, predict_heatmap = net([image, gaze_field, eye_position, head_pose])

            heatmap_loss, m_angle_loss = F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)
            ep_heatmap_loss.append(np.array(heatmap_loss.cpu().data))
            ep_m_angle_loss.append(np.array(m_angle_loss.cpu().data))
            loss = m_angle_loss + heatmap_loss

            loss.backward()
            optimizer.step()

            running_loss.append([heatmap_loss.data, m_angle_loss.data, loss.data])
            # logging.info('%s %s %s'%(str(np.mean(running_loss, axis=0)), method, str(lr_scheduler.get_lr())))
            # if i % 10 == 9:
            #     logging.info('%s %s %s'%(str(np.mean(running_loss, axis=0)), method, str(lr_scheduler.get_lr())))
            #     running_loss = []

        epoch += 1
        print('==== Training loss ====')
        print('Heatmap loss: ', np.mean(np.array(ep_heatmap_loss)))
        print('Mean angle loss: ', np.mean(np.array(ep_m_angle_loss)))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('Saving model to output path: ', output_path+'/epoch_{}_loss_{}.pkl'.format(epoch, loss.data))
        torch.save(net.state_dict(), output_path+'/epoch_{}_loss_{}.pkl'.format(epoch, loss.data))
        print('Starting model test')
        test(net, test_data_loader)


if __name__ == '__main__':
    main()
