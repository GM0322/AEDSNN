import torch
import torch.utils.data as data
import numpy as np
import os

__all__ = ['aeds_loader','fbpconvnet_loader']

class aeds_data(data.Dataset):
    def __init__(self,input_path, label_path,nViews,nBins,image_size):
        self.input_path = input_path
        self.label_path = label_path
        self.nViews = nViews
        self.nBins = nBins
        self.image_size = image_size
        self.files = os.listdir(input_path)

    def __getitem__(self, item):
        input = torch.from_numpy(np.fromfile(self.input_path+'/'+self.files[item],dtype=np.float32)).view(1,self.nViews,self.nBins)
        label = torch.from_numpy(np.fromfile(self.label_path+'/'+self.files[item],dtype=np.float32)).view(1,self.image_size,self.image_size)
        return input,label,self.files[item]

    def __len__(self):
        return len(self.files)

class image_data(data.Dataset):
    def __init__(self,input_path,label_path,image_size):
        self.input_path = input_path
        self.label_path = label_path
        self.image_size = image_size
        self.files = os.listdir(input_path)

    def __getitem__(self, item):
        input = torch.from_numpy(np.fromfile(self.input_path+'/'+self.files[item],dtype=np.float32)).view(1,self.image_size,self.image_size)
        label = torch.from_numpy(np.fromfile(self.label_path+'/'+self.files[item],dtype=np.float32)).view(1,self.image_size,self.image_size)
        return input,label,self.files[item]

    def __len__(self):
        return len(self.files)

def aeds_loader(input_path,nViews,nBins,image_size, batch_size=1,shuffle=True,label_path=None):
    if label_path == None:
        label_path = input_path+'/../label'
    dataset = aeds_data(input_path,label_path,nViews,nBins,image_size)
    loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    return loader

def image_loader(input_path,image_size, batch_size=1,shuffle=True,label_path=None):
    if label_path == None:
        label_path = input_path+'/../label'
    dataset = image_data(input_path,label_path,image_size)
    loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    return loader
