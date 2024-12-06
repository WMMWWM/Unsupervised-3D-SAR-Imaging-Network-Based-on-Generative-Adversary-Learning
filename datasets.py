import glob
import random
import os
import numpy as np
from tqdm import tqdm
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)


def mask_generator(sampling_rate=0.5,model='uniform'):
    '''
    generate a mask with size :512*512 and sampling_rate: 0.5 
    '''
    # axis_all = torch.arange(0,512*512)
    if model == 'block':
        axis_all = torch.randperm(64*64).view(64,64)
        axis_all= (axis_all <sampling_rate*64*64).float()
        axis_all = torch.kron(axis_all, torch.ones((8, 8), dtype=axis_all.dtype))
        return axis_all
    elif model == 'uniform':
        axis_all = torch.randperm(128*128).view(128,128)
        axis_all= (axis_all <sampling_rate*128*128).float()
        axis_all = torch.kron(axis_all, torch.ones((4, 4), dtype=axis_all.dtype))
        return axis_all
    elif model == 'tie':
        axis_all = torch.randperm(512).view(512)
        axis_all = (axis_all < sampling_rate*512).float()
        axis_all = torch.kron(axis_all, torch.ones((1, 512), dtype=axis_all.dtype))
        return axis_all
    elif model == 'easy':
        axis_all = torch.randperm(256*256).view(256,256)
        axis_all = (axis_all <sampling_rate*256*256).float()
        axis_all = torch.kron(axis_all, torch.ones((2, 2), dtype=axis_all.dtype))
    elif model == '1':
        axis_all = torch.randperm(512*512).view(512,512)
        axis_all = (axis_all <sampling_rate*512*512).float()
        return axis_all


class imagenet_dataset(Dataset):
    def __init__(self,dir_path,num):
        self.itername = 'echo_imagenet_%d.mat'
        self.echoname = 'save_mat'
        self.len = num
        self.dir_path = dir_path
        # self.label_path = dir_path.replace('SNR10','SNR25')
        self.label_path = dir_path
        self.label = []
        self.dir = []
        for i in tqdm(range(self.len)):
            labels = h5py.File(os.path.join(self.label_path,self.itername % (i+1)))[self.echoname][:].transpose()
            labels = torch.from_numpy(labels).float()
            input = h5py.File(os.path.join(self.dir_path,self.itername % (i+1)))[self.echoname][:].transpose()
            input = torch.from_numpy(input).float()
            self.label.append(labels)
            self.dir.append(input)
        print('Total %d items are loaded!' % self.len)

    def __getitem__(self, index):
        labels = self.label[index]
        inputs = self.dir[index]
        mask = mask_generator(0.4)
        inputs = inputs*mask
        inputs = inputs.squeeze(0)
        labels = labels.squeeze(0)
        return labels.cuda(),inputs.cuda(),mask.squeeze(0).cuda()
    def __len__(self):
        return self.len
    
def normalize_data(data):
        min_val = torch.min(data)
        max_val = torch.max(data)
        normalized_data = (data - min_val) / (max_val - min_val) * 2 - 1
        return normalized_data

def norm_data(x):
    x_complex = x[:, 0, :, :] + x[:, 1, :, :] * 1j
    x_abs = torch.abs(x_complex).reshape(x.size()[0],1,512,512)
    x_temp = x_abs.reshape(x_abs.size()[0],-1)
    x_max = torch.max(x_temp,dim=-1)[0].reshape(-1,1,1,1)
    x_norm = x/x_max
    x_abs_norm = x_abs/x_max
    return x_norm.cuda(),x_max.cuda(),x_abs_norm

def norm_1d(x):
    x_max = torch.max(x.reshape(x.shape[0],-1),dim=1)[0].reshape(x.shape[0],1,1,1)
    return x/x_max,x_max,x/x_max
def check(x):
    return torch.max(x), torch.min(x)


def from2Dto1D(x):
    x= torch.cat(torch.split(x,split_size_or_sections=1,dim=1),dim=0)
    return x
def from1Dto2D(x):
    x= torch.cat(torch.split(x,split_size_or_sections=4,dim=0),dim=1)
    return x


class Loss_abs(object):
    def __init__(self, pic1, pic2):
        super(Loss_abs,self).__init__()
        self.p1 = pic1
        self.p2 = pic2

    def l1_loss(self):
        return torch.mean(torch.abs(self.p1-self.p2))

    def l2_loss(self):
        return torch.mean((self.p1-self.p2)**2)

    def ssim(self):
        p1 = np.array(self.p1.cpu().detach().numpy()).reshape(512,512)
        p2 = np.array(self.p2.cpu().detach().numpy()).reshape(512,512)
        return SSIM(p1,p2,data_range=1.0)

    def psnr(self):
        p1 = np.array(self.p1.cpu().detach().numpy()).reshape(512,512)
        p2 = np.array(self.p2.cpu().detach().numpy()).reshape(512,512)
        return PSNR(p1,p2,data_range=1.0)
    
def norm2(x):
    x_complex = x[:, 0, :, :] + x[:, 1, :, :] * 1j
    x_abs = torch.abs(x_complex)
    x_abs = x_abs.reshape(x.size()[0],-1)
    x_scale,_ = torch.max(x_abs,dim=-1)
    x_scale = x_scale.reshape(-1,1,1,1)
    x = x / x_scale
    return x_scale, x


class echo_dataloader(Dataset):
    def __init__(self,dir_path,num):
        self.itername = 'echo_imagenet_%d.mat'
        self.echoname = 'input'
        self.gtname = 'gt'
        self.len = num
        self.dir_path = dir_path
        # self.label_path = dir_path.replace('SNR10','SNR25')
        self.label_path = dir_path
        self.label = []
        self.dir = []
        for i in tqdm(range(self.len)):
            labels = h5py.File(os.path.join(self.label_path,self.itername % (i+1)))[self.gtname][:].transpose()
            labels = torch.from_numpy(labels).float()
            input = h5py.File(os.path.join(self.dir_path,self.itername % (i+1)))[self.echoname][:].transpose()
            input = torch.from_numpy(input).float()
            self.label.append(labels)
            self.dir.append(input)
        print('Total %d items are loaded!' % self.len)

    def __getitem__(self, index):
        labels = self.label[index]
        inputs = self.dir[index]
        labels = inputs.squeeze(0)
        mask = mask_generator(0.4)
        scene_mask = torch.zeros(512,512)
        scene_mask[156:156+200,56:56+400] = 1
        # mask = mask*scene_mask
        inputs = inputs*mask
        inputs = inputs.squeeze(0)
        return labels.cuda(),inputs.cuda(),mask.squeeze(0).cuda()
    def __len__(self):
        return self.len
def normalize_data(data):
        min_val = torch.min(data)
        max_val = torch.max(data)
        normalized_data = (data - min_val) / (max_val - min_val) * 2 - 1
        return normalized_data

def norm_data(x):
    x_complex = x[:, 0, :, :] + x[:, 1, :, :] * 1j
    x_abs = torch.abs(x_complex).reshape(x.size()[0],1,512,512)
    x_temp = x_abs.reshape(x_abs.size()[0],-1)
    x_max = torch.max(x_temp,dim=-1)[0].reshape(-1,1,1,1)
    x_norm = x/x_max
    x_abs_norm = x_abs/x_max
    return x_norm.cuda(),x_max.cuda(),x_abs_norm

def norm_1d(x):
    x_max = torch.max(x.reshape(x.shape[0],-1),dim=1)[0].reshape(x.shape[0],1,1,1)
    return x/x_max,x_max,x/x_max
def check(x):
    return torch.max(x), torch.min(x)


def from2Dto1D(x):
    x= torch.cat(torch.split(x,split_size_or_sections=1,dim=1),dim=0)
    return x
def from1Dto2D(x):
    x= torch.cat(torch.split(x,split_size_or_sections=4,dim=0),dim=1)
    return x


class cori_loss(object):
    def __init__(self, pic1, pic2):
        super(cori_loss,self).__init__()
        self.gt = torch.abs(pic1).detach().cpu().squeeze().numpy()
        self.recons = torch.abs(pic2).detach().cpu().squeeze().numpy()

    def l1_loss(self):
        return np.mean(np.abs(self.gt-self.recons))

    def l2_loss(self):
        return np.sqrt(np.sum((self.gt-self.recons)**2))/self.gt.shape[0]

    def ssim(self):
        return SSIM(self.gt,self.recons,data_range=self.gt.max())
    
    def norm(self, x):
        # min-max normalization
        return (x - x.min())/(x.max() - x.min() + 1e-9)
    
    def psnr(self):
        err = self.gt - self.recons
        denom = np.mean(pow(err, 2))
        snrval = 10 * np.log10(np.max(self.gt) / denom)
        return snrval