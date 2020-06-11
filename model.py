import os
import sys

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from PIL import Image
#from dataset import synth_images, real_images
from loss_func import *
from dataset import *
from buffer import Buffer

import numpy as np

#from dlutils import plot_image_batch_w_labels

#from utils.image_history_buffer import ImageHistoryBuffer

syn_dir = 'gaze.h5'
real_dir = 'real_gaze.h5'

# image dimensions
img_width = 55
img_height = 35
img_channels = 1

# training params
nb_steps = 10000
bs = 256
K_d = 1  # number of discriminator updates per step
K_g = 2  # number of generative network updates per step
log_interval = 100

n=0.01
Buffer = Buffer()


class R(nn.Module):
    """
     A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
     each with `nb_features` feature maps.
     See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
     :param input_features: Input tensor to ResNet block.
     :return: Output tensor from ResNet block.
    """
    def __init__(self):
        super(R, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3,padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, 3,padding=1)
        self.conv4 = nn.Conv2d(64,1,1)
        self.tanh = nn.Tanh()


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        for _ in range(4):
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
        x = self.conv4(x)
        x = self.tanh(x)

        return x


class D(nn.Module):
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.
    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    def __init__(self,):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 64, 3, stride=2, padding=1)
        self.maxpool2d = nn.MaxPool2d(3, stride=1)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 2, 1, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool2d(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.softmax(x)
        return x





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ref_loss=[]
dis_loss=[]
#y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
#y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)


print('---Train Refiner Network 1000 times---')
synth_images = synth_images(syn_dir,bs)
real_images = real_images(real_dir,bs)



RefNet = R()
DisNet = D()
RefNet.to(device)
DisNet.to(device)

optimizer_ref = optim.SGD(RefNet.parameters(), lr=0.001)

running_loss = 0
R_loss = 0
D_loss = 0

for i in range(50):
    synth_batch = iter(synth_images).next()
    synth_batch = synth_batch.to(device)
    RefNet.train()
    optimizer_ref.zero_grad()
    R_output = RefNet(synth_batch)
    DisNet.eval()
    D_real_out = DisNet(R_output)
    if i > 10:
        Buffer.add_to_buffer(R_output.cpu().data.numpy())
    r_loss_ref = self_reg_loss(n, R_output, synth_batch)
    r_loss_realism = local_adv_loss(D_real_out, real_label(D_real_out))
    print(r_loss_ref)
    print(r_loss_realism)
    R_loss = (r_loss_ref + r_loss_realism)
    print(R_loss)
    R_loss.backward()
    optimizer_ref.step()

    running_loss += R_loss.item()
#    ref_loss.append(running_loss)
    if i % 10==0:
        print('Ref loss:{} ,Epochs:{} '.format(running_loss/(i+1), i+1))





#assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)
print('---Train Discriminator Network 200 times---')


optimizer_dis = optim.SGD(DisNet.parameters(), lr=0.001)
DisNet.to(device)
running_loss = 0

for _ in range(20):
    RefNet.eval()
    DisNet.train()
    optimizer_dis.zero_grad()

    synth_batch = iter(synth_images).next()
    synth_batch = synth_batch.to(device)
    real_batch = iter(real_images).next()
    real_batch = real_batch.to(device)

    synth_batch = synth_batch.to(device)
    real_batch = real_batch.to(device)

    ref_batch = RefNet(synth_batch)
    batch_from_buffer = Buffer.get_from_buffer()
    batch_from_buffer = torch.from_numpy(batch_from_buffer)
    ref_batch[:bs//2] = batch_from_buffer

    D_real_out = DisNet(real_batch)#.view(-1,2)
    D_loss_real = local_adv_loss(D_real_out, real_label(D_real_out))

    D_ref_out = DisNet(ref_batch)#.view(-1,2)
    D_loss_ref = local_adv_loss(D_ref_out, fake_label(D_ref_out))

    D_loss = (D_loss_ref + D_loss_real)
    D_loss.backward()
    optimizer_dis.step()

    running_loss += D_loss.item()
#    dis_loss.append(running_loss)
    if i % 2==0:
        print('Dis loss:{} ,Epochs:{} '.format(running_loss/(i+1), i+1))


T=5

for i in range(T):
    R_loss, D_loss = 0, 0
    R_epoch_loss, D_epoch_loss = 0, 0
    for _ in range(K_g):

        synth_batch = iter(synth_images).next()
        synth_batch = synth_batch.to(device)
        RefNet.train()
        optimizer_ref.zero_grad()
        R_output = RefNet(synth_batch)
        DisNet.eval()
        D_real_out(R_output)

        Buffer.add_to_buffer(R_output.cpu().data.numpy())

        r_loss_ref = self_reg_loss(n, R_output, synth_batch)
        r_loss_realism = local_adv_loss(D_real_out, real_label(D_real_out))
        R_loss = (r_loss_ref + r_loss_realism)
        R_epoch_loss += R_loss.item()
        R_loss.backward()
        optimizer_ref.step()



    for _ in range(K_d):

        synth_batch = iter(synth_images).next()
        synth_batch = synth_batch.to(device)
        real_batch = iter(real_images).next()
        real_batch = real_batch.to(device)

        Refnet.eval()
        ref_batch = RefNet(synth_batch)

        batch_from_buffer = Buffer().get_from_buffer()
        Buffer.add_to_buffer(synth_batch.cpu().data.numpy())
        #numpy <--> torch.tensor
        batch_from_buffer = torch.from_numpy(batch_from_buffer)
        ref_batch[:bs//2] = batch_from_buffer

        D_real_out = DisNet(real_batch)#.view(-1,2)
        D_loss_real = local_adv_loss(D_real_out, real_label(D_real_out))

        D_ref_out = DisNet(ref_batch)#.view(-1,2)
        D_loss_ref = local_adv_loss(D_ref_out, fake_label(D_ref_out))
        D_loss = (D_loss_ref + D_loss_real)
        D_epoch_loss += D_loss.item()
        D_loss.backward()
        optimizer_dis.step()

    if i % log_interval:
        print('Refiner model loss: {}.'.format(R_epoch_loss / K_g))
        print('Discriminator model loss: {}.'.format(D_epoch_loss / K_d))
        #print('Discriminator model loss refined: {}.'.format(D_loss_ref))

        fig = plt.figure(figsize=(55,35))
        rows = 2
        columns = 10
        syn_imgs = ref_batch[train_bs//2:train_bs//2+10]
        syn_imgs = syn_imgs.permute(0,2,3,1).cpu().data.numpy()
        ref_imgs = R_output[bs//2:bs//2+10]
        ref_imgs = ref_imgs.permute(0,2,3,1).cpu().data.numpy()
        for i in range(10):
            syn = syn_imgs[i]
            ref = ref_imgs[i]
            ax = plt.subplot(rows, columns,i+1)
            ax.imshow(syn)
            ax.axis('off')
            ax = plt.subplot(rows, columns, i+11)
            ax.imshow(ref)
            ax.axis('off')
        plt.show()
