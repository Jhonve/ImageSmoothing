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

from tensorboardX import SummaryWriter

k_opt = getParser()
k_epoch = k_opt.num_epoch
k_lambda_flat = 8.0
k_lambda_edge = 24.0

k_loss_writer = SummaryWriter("runs/losses")

if not os.path.exists(k_opt.val_path):
    os.makedirs(k_opt.val_path)

if not os.path.exists(k_opt.ckpt_path):
    os.makedirs(k_opt.ckpt_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

def splitData(data_path, num_val_batch):
    num_data = data_path.shape[0]
    num_val_data = num_val_batch * k_opt.batch_size
    num_train_data = num_data - num_val_data

    val_index = random.sample(range(0, num_data), num_val_data)
    train_index = list(set(range(0, num_data)) - set(val_index))

    train_path = data_path[train_index]
    val_path = data_path[val_index]

    val_index = np.array(val_index)
    np.save(k_opt.val_path + "val_index.npy", val_index)

    return train_path, val_path

def reSplitData(data_path):
    num_data = data_path.shape[0]
    val_index = np.load(k_opt.val_res_path + "val_index.npy")
    val_index = list(val_index)
    train_index = list(set(range(0, num_data)) - set(val_index))

    train_path = data_path[train_index]
    val_path = data_path[val_index]

    return train_path, val_path

def train():
    data_path_file = k_opt.data_path_file
    data_path = h5py.File(data_path_file, 'r')
    data_path = np.array(data_path["data_path"])

    if k_opt.current_model != "":
        train_path, val_path = reSplitData(data_path)
    else:
        train_path, val_path = splitData(data_path, k_opt.num_val_batch)
    num_train_batch = int(train_path.shape[0] / k_opt.batch_size)

    # initialize Dataloader
    train_dataset = ImgDataset(k_opt, train_path, 224, is_train=True)
    train_data_loader = train_dataset.getDataloader()

    val_dataset = ImgDataset(k_opt, val_path, 224, is_train=False)
    val_data_loader = val_dataset.getDataloader()

    # initialize Network structure etc.
    current_epoch = 0
    smooth_net = SmoothNet(3, 3)
    smooth_net = torch.nn.DataParallel(smooth_net)
    if k_opt.current_model != "":
        smooth_net.load_state_dict(torch.load(k_opt.current_model))
        print("Load ", k_opt.current_model, " Success!")
        current_epoch = int(k_opt.current_model.split('/')[-1].split('_')[0]) + 1
    optimizer = Toptim.Adam(smooth_net.parameters(), lr=k_opt.learning_rate, betas=(0.9, 0.999))
    smooth_net.cuda()

    gaussian_kernel = kornia.filters.GaussianBlur2d((21, 21), (7, 7))
    laplace_kernel = kornia.filters.Laplacian(3)

    for epoch in range(current_epoch, k_epoch):
        smooth_net = smooth_net.train()
        for i_train, data in enumerate(train_data_loader, 0):
            input_img = data
            input_img = input_img.type(torch.FloatTensor)
            input_img = input_img.cuda()
            input_img = input_img.permute(0, 3, 1, 2)

            input_img_copy = input_img.clone()

            optimizer.zero_grad()
            output = smooth_net(input_img)

            loss_data = F.mse_loss(output, input_img_copy)
            
            # edge loss
            edge_input = kornia.filters.Sobel()(input_img_copy)
            edge_map = edge_input > 0.08
            edge_map = edge_map.type(torch.FloatTensor).cuda()
            loss_edge = F.mse_loss(edge_map * output, edge_map * input_img_copy)

            # laplace_input = laplace_kernel(input_img_copy)

            # flat_loss
            output_copy = output.clone()
            flat_output = gaussian_kernel(output_copy)
            flat_map = (edge_map - 1.) * -1.
            loss_flat = F.mse_loss(flat_map * output, flat_map * flat_output)

            loss = loss_data + k_lambda_flat * loss_flat + k_lambda_edge * loss_edge
            
            if(i_train % 30 == 0):
                k_loss_writer.add_scalar('train_loss', loss, global_step=epoch * num_train_batch + i_train + 1)
                k_loss_writer.add_scalar('train_loss_data', loss_data, global_step=epoch * num_train_batch + i_train + 1)
                k_loss_writer.add_scalar('train_loss_flat', loss_flat, global_step=epoch * num_train_batch + i_train + 1)
                k_loss_writer.add_scalar('train_loss_edge', loss_edge, global_step=epoch * num_train_batch + i_train + 1)

            loss.backward()
            optimizer.step()

            print("Epcoh: %d, || Batch: %d/%d, || current loss: %.7f, data loss: %.7f, flat loss: %.7f, edge loss: %.7f" %
                (epoch, i_train + 1, num_train_batch, loss.data.item(), loss_data.data.item(), loss_flat.data.item(), loss_edge.data.item()))
        
        # ____Validation____
        torch.save(smooth_net.state_dict(), k_opt.ckpt_path + str(epoch) + "_model.t7")
        val_loss = []
        val_loss_data = []
        val_loss_flat = []
        val_loss_edge = []
        shown_input_imgs = []
        shown_output_imgs = []
        smooth_net = smooth_net.eval()
        for i_val, data in enumerate(val_data_loader, 0):
            input_img = data

            input_img = input_img.type(torch.FloatTensor)
            input_img = input_img.cuda()
            input_img = input_img.permute(0, 3, 1, 2)
            shown_input_imgs.append(input_img[0].detach().cpu().numpy())

            input_img_copy = input_img.clone()
            laplace_input = laplace_kernel(input_img_copy)
            output = smooth_net(input_img)

            shown_output = output[0].clone()
            shown_output = shown_output - laplace_input[0]
            shown_output_imgs.append(shown_output.detach().cpu().numpy())

            loss_data = F.mse_loss(output, input_img_copy)
            
            # edge loss
            edge_input = kornia.filters.Sobel()(input_img_copy)
            edge_map = edge_input > 0.08
            edge_map = edge_map.type(torch.FloatTensor).cuda()
            loss_edge = F.mse_loss(edge_map * output, edge_map * input_img_copy)

            # flat_loss
            output_copy = output.clone()
            flat_output = gaussian_kernel(output_copy)
            flat_map = (edge_map - 1.) * -1.
            loss_flat = F.mse_loss(flat_map * output, flat_map * flat_output)

            # loss = loss_data + k_lambda_flat * loss_flat + k_lambda_edge * loss_edge
            loss = k_lambda_flat * loss_flat + k_lambda_edge * loss_edge

            val_loss.append(loss.data.item())
            val_loss_data.append(loss_data.data.item())
            val_loss_flat.append(loss_flat.data.item())
            val_loss_edge.append(loss_edge.data.item())

            print("Epcoh: %d, || Val Batch: %d/%d, || current loss: %.7f, data loss: %.7f, flat loss: %.7f, edge loss: %.7f" % \
                (epoch, i_val + 1, k_opt.num_val_batch, loss.data.item(), loss_data.data.item(), loss_flat.data.item(), loss_edge.data.item()))

        val_loss = np.array(val_loss)
        val_loss_data = np.array(val_loss_data)
        val_loss_flat = np.array(val_loss_flat)
        val_loss_edge = np.array(val_loss_edge)

        mean_loss = np.mean(val_loss)
        mean_loss_data = np.mean(val_loss_data)
        mean_loss_flat = np.mean(val_loss_flat)
        mean_loss_edge = np.mean(val_loss_edge)
        
        k_loss_writer.add_scalar('val_loss', mean_loss, global_step=epoch + 1)
        k_loss_writer.add_scalar('val_loss_data', mean_loss_data, global_step=epoch + 1)
        k_loss_writer.add_scalar('val_loss_flat', mean_loss_flat, global_step=epoch + 1)
        k_loss_writer.add_scalar('val_loss_edge', mean_loss_edge, global_step=epoch + 1)

        shown_input_imgs = np.array(shown_input_imgs)
        shown_output_imgs = np.array(shown_output_imgs)

        k_loss_writer.add_images('val_input_imgs', shown_input_imgs, global_step=epoch + 1)
        k_loss_writer.add_images('val_output_imgs', shown_output_imgs, global_step=epoch + 1)

if __name__ == "__main__":
    train()
