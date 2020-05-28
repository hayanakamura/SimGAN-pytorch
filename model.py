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

def train_fn(net, loader):
    running_loss = 0
    runnning_corrects = 0
    preds_for_acc = []
    pbar = tqdm(total = len(loader), desc='Training')

    for _, (images, labels) in enumerate(loader):

        images = images.to(device)
        net.train()
        optimizer.zero_grad()
        predictions = net(images)
        print(predictions,labels)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
        preds = np.argmax(predictions.cpu().detach().numpy(), 1)
        preds_for_acc = np.concatenate((preds_for_acc, preds), 0)
        pred = torch.from_numpy(preds)
        runnning_corrects += torch.sum(pred==labels)
        pbar.update()

    #accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    accuracy = runnning_corrects.item() / dataset_sizes['train']

    pbar.close()
    return running_loss/dataset_sizes['train'], accuracy


#R_output = R(image)
#D_output = D(R_output)

RefNet = R()
DisNet = D()
loss = 

print('---Train Refiner Network 1000 times---')
synth_images = synth_images(imgdir,bs)

for _ in range(1000):
    for images in synth_images:
        images = images.to(device)
        RefNet.train()
        optimizer.zero_grad()
        R_output = RefNet(images)
        loss =

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
