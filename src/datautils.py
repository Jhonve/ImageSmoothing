import os
import random

import numpy as np
import h5py
import cv2

import torch
import torch.utils.data as Tdata

from parsers import getParser

k_opt = getParser()

class ImgDataset(Tdata.Dataset):
    def __init__(self, parser, data_path, img_size, is_train):
        super(ImgDataset).__init__()
        self.batch_size = parser.batch_size
        self.num_workers = parser.num_workers
        self.data_path = data_path
        self.img_size = img_size
        self.SIZE = len(data_path)

        self.is_train = is_train

    def __len__(self):
        return self.SIZE

    def loadImg(self, data_path):
        image_data = cv2.imread(data_path)
        image_data = np.array(image_data).astype(np.float)
        image_data = image_data / 255.

        width = image_data.shape[0]
        height = image_data.shape[1]
        if(width >= height):
            image_data = image_data[0:height, : , :]
        else:
            image_data = image_data[:, 0:width, :]
        image_data = cv2.resize(image_data, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        return image_data

    def __getitem__(self, index):
        input_img = self.loadImg(self.data_path[index])
        return input_img

    def getDataloader(self):
        if(self.is_train):
            return Tdata.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        else:
            return Tdata.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)

def preDataPath(folder_list):
    all_file_list = []
    for i in range(len(folder_list)):
        file_list = os.listdir(k_opt.data_path + folder_list[i])
        for j in range(len(file_list)):
            if(file_list[j][-5] == '!' or file_list[j][-6] == '!'):
                continue
            data_path = k_opt.data_path + folder_list[i] + '/' + file_list[j]
            image = cv2.imread(data_path)
            try:
                image.shape
            except:
                # print('image read error')
                continue
            file_list[j] = k_opt.data_path + folder_list[i] + '/' + file_list[j]
            all_file_list.append(file_list[j])
            print(i, j)

    return all_file_list

def saveH5(files_paths):
    files_paths = np.array(files_paths)
    
    with h5py.File(k_opt.data_path_file, 'w') as data_file:
        data_type = h5py.special_dtype(vlen=str)
        data = data_file.create_dataset("data_path", files_paths.shape, dtype=data_type)
        data[:] = files_paths
        data_file.close()

    print("Save path done!")

def selfTest():
    '''
    # initialize data path to datapath.h5
    folder_list = os.listdir(k_opt.data_path)
    print("Number of folders: ", len(folder_list))

    files_paths = preDataPath(folder_list)
    print("Number of data: ", len(files_paths))
    saveH5(files_paths)
    '''

    data_path_file = k_opt.data_path_file
    data_path = h5py.File(data_path_file, 'r')

    data_path = np.array(data_path["data_path"])
    dataset = ImgDataset(k_opt, data_path, 224, is_train=False)
    data_loader = dataset.getDataloader()

    for i, data in enumerate(data_loader, 0):
        input_img = data
        print(input_img.shape, i)
    
if __name__ == "__main__":
    selfTest()