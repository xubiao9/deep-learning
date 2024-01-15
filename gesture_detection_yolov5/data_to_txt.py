import os
import torch
import random
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import math

img_path = 'E:/!_AI_self_Proj/Gesture_Detection_Yolov5/dataset/hand_gesture_dataset/images/'
label_path = 'E:/!_AI_self_Proj/Gesture_Detection_Yolov5/dataset/hand_gesture_dataset/labels/'

train_len = len(os.listdir(img_path+'train'))
test_len = len(os.listdir(img_path+'val'))
print(train_len)

t_or_l = ['train', 'val']

# 写train文件
for t_or_l in t_or_l:
    print(t_or_l)
    if t_or_l == 'train':
        len1 = 0
        len2 = train_len
        t = len2
    else:
        len1 = t
        len2 = t+test_len

    list_file = open('%s_%s.txt' % ('gesture', t_or_l), 'w', encoding='utf-8')
    for i in range(len1, len2):
        # print(i)
        idx = str(i).zfill(4)
        path = label_path+t_or_l

        # 打开标签文件
        txtFile = open(os.path.join(path+'/', str(idx)+'.txt'), encoding='utf-8').read().strip().split()
        # 图片格式：(960, 1280, 3)
        # 从（类别，中心x比例，中心y比例，w比例，h比例）变成 （x1，y1，x2，y2）
        w = 1280
        h = 960
        [x1, y1, x2, y2] = [int(float(txtFile[1])*w)-int(float(txtFile[3])*w)//2, int(float(txtFile[2])*h)-int(float(txtFile[4])*h)//2,
                            int(float(txtFile[1])*w)+int(float(txtFile[3])*w)//2, int(float(txtFile[2])*h)+int(float(txtFile[4])*h)//2]
        list_file.write(img_path+t_or_l+'/'+str(idx)+'.png') # 写图片路径
        list_file.write(" " + ",".join([str(j) for j in [x1, y1, x2, y2]]) + ',' + str(txtFile[0])) # 写标签
        list_file.write("\n")
    list_file.close()