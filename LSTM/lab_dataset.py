import os
import queue as Queue
import threading

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import functional as F
import pickle
from PIL import Image


class motionset(Dataset):
    def __init__(self,pkl,set_type,feature_szie):
        super(motionset,self).__init__()
        self.pkl = pkl
        self.set_type = set_type
        with open(pkl,'rb') as f:
            self.items = pickle.load(f)
        self.feature_size = feature_szie

    def to_one_hot(self,label,dimension=6):
        result = np.zeros(dimension)
        result[int(label)-1] = 1
        result = result.astype('float64')
        return result 
        
        
    def __getitem__(self, index):
        ### {'data':data,'label':label}
        item = self.items[index]
        data = item['data']
        # label = self.to_one_hot(item['label'])
        label = item['label']-1
        data = torch.tensor(data,dtype=torch.float64)
        label = torch.tensor(label,dtype=torch.float64)
        return data,label
    
    def __len__(self,):
        return len(self.items)


