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
import cv2
from PIL import Image
import dlib
from imutils import face_utils
from sklearn import preprocessing

from utils import data_transforms

from headPose.detectHeadpose import HeadPoseEstimation


# landmarks_predictor_model = 'model/shape_predictor_68_face_landmarks.dat'
# predictor = dlib.shape_predictor(landmarks_predictor_model)
# face_detector = cv2.CascadeClassifier('model/lbpcascade_frontalface_improved.xml')
headpose = HeadPoseEstimation(0, 1)
min_max_scaler = preprocessing.MinMaxScaler()

def generate_data_field(eye_point):
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

def test(net, originalImage, eye, headpose):
    net.eval()
    heatmaps = []

    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    # generate gaze field
    gaze_field = generate_data_field(eye_point=eye)
    TorchGaze_field = torch.from_numpy(gaze_field)
    head_pose = torch.FloatTensor(headpose)
    eye_position = torch.FloatTensor(eye)

    image, gaze_field, eye_position, head_pose = map(lambda x: Variable(x.unsqueeze(0).cuda(), volatile=True), [image, TorchGaze_field, eye_position, head_pose])

    _, predict_heatmap = net([image, gaze_field, eye_position, head_pose])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])

    return heatmap, f_point[0], f_point[1] 

def draw_result(im, eye, heatmap, gaze_point):
    x1, y1 = eye
    x2, y2 = gaze_point
    image_height, image_width = im.shape[:2]
    x1 = image_width * x1
    y1 = y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 3)
    
    # heatmap visualization
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = cv2.resize(heatmap, (image_width, image_height))

    heatmap = (0.8 * heatmap.astype(np.float32) + 0.2 * im.astype(np.float32)).astype(np.uint8)
    img = np.concatenate((im, heatmap), axis=1)
    # cv2.imwrite('tmp.png', img)
    
    return img

def detect_eyes(image, faces):
    rpta = []
    if len(rects) > 0:
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            rpta = shape[27]   
        return rpta
    else:
        return []

def detect_eyes2(gray):
    rects = detector(gray, 1)
    rpta = []
    if len(rects) > 0:
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            rpta = shape[27]   
        return rpta
    else:
        return []

def deepMain():
    # from deepface.detectors.detector_ssd import FaceDetectorSSDMobilenetV2

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()
    face_detector = FaceDetectorSSDMobilenetV2()

    # Load pretrained gaze following model
    pretrained_dict = torch.load('/home/emannuell/Documentos/mestrado/GazeFollowing/model/test04/epoch_15_loss_0.0558342523873.pkl')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    cap = cv2.VideoCapture('/home/emannuell/Documentos/mestrado/renault/GOPR1503.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(width, height, videoFrames, fps)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # out = cv2.VideoWriter('inspecaoCarroResult.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    print('Iniciando processamento do video...')
    while True:
        frameId = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frameId < videoFrames:
            print(frameId,'/',videoFrames)
            ret, img = cap.read()
            t = time.time()
            originalImg = img
            # img = cv2.resize(img, (640, 480))
            # height, width, _ = img.shape
            faces = face_detector.detectMultiScale(img, 1.05, 3)
            for face in faces:
                print(face)
            # Precisa redimensionar imagem para aumentar o crop!
            # faceImage = img[face[1]:face[1] + face[2], face[0]:face[3] + face[0]]
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # formatDetected = [(face[1],face[1] + face[2]), (face[0],face[3] + face[0])]
            # print(formatDetected)
            # x_ = face[0]
            # y_ = face[1]
            # w_ = face[2]
            # h_ = face[3]
            # score = face.score
            # x, y = int(x_ + w_ / 2), int(y_ + h_ /2)
            # print(x, y)
            # cv2.rectangle(img, (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 2)
            # Head pose estimation
            # faceBbox = x_, y_, x_ + w_, y_ + h_
            # y, p, r = headpose.detectHeadPose(img, faceBbox)
            # To normalize detections we fit with min and max angular examples from dataset
            # MinMaxY = [-86.73414612, 87.62715149]
            # MinMaxP = [-54.0966301, 36.50032043]
            # MinMaxR = [-42.67565918, 42.16217041]
            # MinMaxY.append(y)
            # MinMaxP.append(p)
            # MinMaxR.append(r)
            # y = min_max_scaler.fit_transform(np.array(MinMaxY).reshape((-1, 1)))
            # p = min_max_scaler.fit_transform(np.array(MinMaxP).reshape((-1, 1)))
            # r = min_max_scaler.fit_transform(np.array(MinMaxR).reshape((-1, 1)))
            # headPoseAngles = float(y[-1]), float(p[-1]), float(r[-1])
            # print('Head pose normalized: ', headPoseAngles)
            # center_x = eye_center[0] / width
            # center_y = eye_center[1] / height
            # img = cv2.circle(img, (x, y), 2, (255, 0, 255), thickness=2) # Magento
            # heatmap, p_x, p_y = test(net, originalImg, (x, y), headPoseAngles)
            # img = cv2.circle(img, (int(p_x * width), int(p_y * height)), 2, (255, 0, 0), thickness=2) # Azul
            # img = draw_result(img, (center_x, center_y), heatmap, (p_x, p_y))
        else:
            break
        # img = np.concatenate((img, img2), axis=1)
        # img2 = img
        # Write the frame into the file 'output.avi'
        # img = cv2.resize(img, (1280, 480))
        # out.write(img)
        # cv2.imshow('result', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    # out.release()
    # cv2.destroyAllWindows()


# CASCADE FACE DETECTOR
# cascPath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# faces = faceCascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5, minSize=(1,1), flags = cv2.CASCADE_SCALE_IMAGE )

if __name__ == '__main__':
    deepMain()
