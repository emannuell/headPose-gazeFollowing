import sys
import numpy as np
import cv2

import torch
# import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import headPose.hopenet as hopenet

class HeadPoseEstimation():

    def __init__(self, gpu, batch_size):
        cudnn.enabled = True
        self.gpu = gpu
        snapshot_path = 'headPose/hopenet_robust_alpha1.pkl'
        
        # ResNet50 structure
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        print ('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        self.model.load_state_dict(saved_state_dict)

        print ('Loading data.')

        self.transformations = transforms.Compose([transforms.Scale(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.model.cuda(gpu)

        print ('Ready to test network.')

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(gpu)
    
    def detectHeadPose(self, image, bbox):
        '''
            Image: CV2 load image imread
            bbox: Integer array len(4), Face detection bbox annotation, like (x1, y1, x2, y2)
        '''
        # Read image
        # frame = cv2.imread(image)
        # 297.10608   80.85623   414.11298   274.99792
        # print(image, bbox)
        frame = image
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        x_min -= 50
        x_max += 50
        y_min -= 50
        y_max += 30
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        # line = 'test/00000004/00004782.jpg,4782-10,0.43,0.08,0.57,0.91,0.72545,0.27455,0.53542,0.40156,coco_val,COCO_val2014_000000048739.jpg'
        height, width, channels = frame.shape

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Crop face loosely
        img = cv2_frame[y_min:y_max,x_min:x_max]
        img = Image.fromarray(img)

        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(self.gpu)

        yaw, pitch, roll = self.model(img)
        
        yaw_predicted = F.softmax(yaw, dim=1)
        pitch_predicted = F.softmax(pitch, dim=1)
        roll_predicted = F.softmax(roll, dim=1)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99
        # print(yaw_predicted, pitch_predicted, roll_predicted.item())
        # Print new frame with cube and axis
        # txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
        # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
        # utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
        # Plot expanded bounding box
        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
        # idx += 1
        # cv2.imwrite('result.jpg', frame)
        # cv2.imshow('img', frame)
        # cv2.waitKey(0)
        return yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item()

# if __name__ == '__main__':
#     image = 'filme.jpg'
#     image = cv2.imread(image)
#     bbox = 297, 80, 414, 274
#     headpose = HeadPoseEstimation(0, 1)
#     y, p, r = headpose.detectHeadPose(image, bbox)
#     print(y, p, r)