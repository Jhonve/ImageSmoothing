import os

import torch
import torch.optim as Toptim
import torch.utils.data
import torch.nn.functional as F
import kornia

import h5py
import numpy as np
import random

from parsers import getParser
from model import SmoothNet
from datautils import ImgDataset
from utils import readSingleImg, saveImg

k_opt = getParser()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

def getTestList(test_folder):
    test_file_list = os.listdir(test_folder)
    print("Number of test images: ", len(test_file_list))

    test_img_list = []
    for i in range(len(test_file_list)):
        input_img = readSingleImg(test_folder + test_file_list[i])
        test_img_list.append(input_img)

    return test_img_list

def test():
    smooth_net = SmoothNet(3, 3)
    smooth_net = torch.nn.DataParallel(smooth_net)
    if k_opt.current_model == "":
        print("No pre-trained model!")
        exit(0)
    else:
        smooth_net.load_state_dict(torch.load(k_opt.current_model))
        print("Load ", k_opt.current_model, " Success!")
    smooth_net.cuda()
    smooth_net.eval()

    laplace_kernel = kornia.filters.Laplacian(3)

    output_path = k_opt.test_path + "/res/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_list = getTestList(k_opt.test_path + "/inputs/")
    num_test = len(test_list)

    for i in range(num_test):
        input_img = test_list[i]
        input_img = input_img.type(torch.FloatTensor)
        input_img = input_img.cuda()
        input_img = input_img.permute(0, 3, 1, 2)
        
        output = smooth_net(input_img)
        laplace_input = laplace_kernel(input_img)

        output = output - 3 * laplace_input
        output_img = output.permute(0, 2, 3, 1).detach().cpu().numpy()
        output_img = output_img[0]
        temp_output_path = output_path + "res_" + str(i) + ".jpg"
        saveImg(temp_output_path, output_img)

        print(i + 1, '/', num_test)

        input_img = None
        output = None
        output_img = None

if __name__ == "__main__":
    test()