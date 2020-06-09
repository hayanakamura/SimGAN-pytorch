import torch
import torch.nn
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import h5py

syn_dir = 'gaze.h5'
real_dir = 'real_gaze.h5'


#norm_stack = lambda x: np.clip((x-127.0)/127.0, -1, 1)
def norm_stack(x):
    # calculate statistics on first 20 points
    mean = np.mean(x[:20])
    std = np.std(x[:20])
    return (1.0*x-mean)/(2*std)




class synth_Datasets(torch.utils.data.Dataset):
    def __init__(self,syndir,tfms):
        self.syndir = syndir
        self.tfms = tfms

        with h5py.File(self.syndir,'r') as t_file:
            self.len = len(t_file['image'])
            assert 'image' in t_file, "Images are missing"
            assert 'look_vec' in t_file, "Look vector is missing"
            assert 'path' in t_file, "Paths are missing"
            self.stack = norm_stack(np.expand_dims(np.stack([a for a in t_file['image'].values()],0), -1))
            #print(syn_image_stack.shape, 'loaded')


    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        img = self.stack[idx,:,:,0]
        img = np.asarray(img)
        img = Image.fromarray(img)

        if self.tfms:
            img = self.tfms(img)

        return img

class real_Datasets(torch.utils.data.Dataset):
    def __init__(self,realdir,tfms):
        self.realdir = realdir
        self.tfms = tfms
        with h5py.File(self.realdir,'r') as t_file:
            self.len = len(t_file['image'])
            assert 'image' in t_file, "Images are missing"
            assert 'look_vec' in t_file, "Look vector is missing"
            assert 'path' in t_file, "Paths are missing"
            self.stack = np.expand_dims(np.stack([a for a in t_file['image'].values()],0), -1)
            #print(syn_image_stack.shape, 'loaded')

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        img = self.stack[idx,:,:,0]
        img = np.asarray(img)
        img = Image.fromarray(img)

        if self.tfms:
            img = self.tfms(img)

        return img



def synth_images(imgdir,bs):

    tfms = transforms.Compose([transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])])
    synth_ds = synth_Datasets(imgdir, tfms)
    synth_dl = torch.utils.data.DataLoader(synth_ds, bs, shuffle=True)
    return synth_dl

def real_images(imgdir,bs):

    tfms = transforms.Compose([transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])])
    real_ds = real_Datasets(imgdir,tfms)
    real_dl = torch.utils.data.DataLoader(real_ds, train_bs, shuffle=True)
    return real_dl
