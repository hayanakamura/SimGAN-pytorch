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
#from loss_func import self_reg_loss, local_adv_loss

import numpy as np

#from dlutils import plot_image_batch_w_labels

#from utils.image_history_buffer import ImageHistoryBuffer

# image dimensions
img_width = 55
img_height = 35
img_channels = 1

# training params
nb_steps = 10000
batch_size = 512
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100




class R(nn.Module):
    """
     A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
     each with `nb_features` feature maps.
     See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
     :param input_features: Input tensor to ResNet block.
     :return: Output tensor from ResNet block.
    """
    def __init__(self):
        super(resnet_block, self).__init__()
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
        self.conv1 = nn.Conv2d(1, 96, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 64, 3, stride=2, padding=1)
        self.maxpool2d = nn.MaxPool2d(3, stride=1)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 2, 1, stride=1, padding=1)
        self.softmax = nn.Softmax()

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


print('---Train Refiner Network 1000 times---')
synth_images = synth_images(imgdir,bs)

RefNet = R()
optimizer_ref = optim.SGD(RefNet.parameters(), lr=0.001)
RefNet.to(device)
running_loss = 0
for i in range(1000):
    synth_batch = iter(synth_images).next()
    synth_batch = synth_batch.to(device)
    RefNet.train()
    optimizer_res.zero_grad()
    R_output = RefNet(synth_batch)
    r_loss_ref = self_reg_loss(n, R_output, images)
    r_loss_ref.backward()
    optimizer.step()

    running_loss += loss.item()
    ref_loss.append(running_loss)
    if i % 50:
        print('Ref loss:{} ,Epochs:{} '.format(running_loss, i))




y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)
print('---Train Discriminator Network 200 times---')
synth_images = synth_images(imgdir,bs)
real_images = real_images(imgdir,bs)

DisNet = D()
optimizer_dis = optim.SGD(DisNet.parameters(), lr=0.001)
DisNet.to(device)
running_loss = 0
for _ in range(200):
    RefNet.eval()
    DisNet.train()
    optimizer_dis.zero_grad()

    synth_batch = iter(synth_images).next()
    synth_batch = synth_batch.to(device)
    real_batch = iter(real_images).next()
    real_batch = real_batch.to(device)

    synth_batch = synth_batch.to(device)
    real_batch = real_batch.to(device)

    D_real_out = DisNet(real_batch)
    d_loss_real = local_adv_loss(D_real_out, y_real)

    ref_batch = RefNet(synth_batch)
    D_ref_out = DisNet(ref_batch)
    d_loss_ref = local_adv_loss(D_ref_out, y_ref)


    optimizer_dis.zero_grad()
    optimizer_dis.step()

    running_loss += loss.item()
    dis_loss.append(running_loss)
    if i % 10:
        print('Dis loss:{} ,Epochs:{} '.format(running_loss, i))

K_g =
K_d =


for i in range():

    for _ in range(K_g):

img_path='mynumber.png'
image=Image.open(img_path)
tfms=transforms.Compose([transforms.Grayscale(),
                        transforms.RandomResizedCrop((55,35))
                        ])
image=tfms(image)
imnp=np.array(image)
image=torch.from_numpy(imnp)
image=torch.reshape(image,(1,1,55,35))

print(image.size())
image=image.float()
net=resnet_block()
out=refiner_network(net,image)
print(out.size())
