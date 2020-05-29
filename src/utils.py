import cv2
import numpy as np

import torch

def readSingleImg(data_path):
    img = cv2.imread(data_path)
    img = np.array(img)
    width = img.shape[0]
    height = img.shape[1]
    if(width % 2 == 1):
        img = img[0:width-1, :, :]
    if(height % 2 == 1):
        img = img[:, 0:height-1, :]

    img = img / 255.
    img = np.expand_dims(img, 0)
    input_img = torch.tensor(img)
    return input_img

def saveImg(data_path, img):
    img = np.clip(img, 0., 1.)
    img = img * 255.
    cv2.imwrite(data_path, img)