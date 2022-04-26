import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import RandomAffine
import torchvision
import time
from PIL import Image


def save_model(model,optim,epoch,path):
    if hasattr(model,"module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    torch.save(model_state,path)

def load_model(model,ckp_path):
    def remove_module_string(k):
        items = k.split(".")
        items = items[0:1] + items[2:]
        return ".".join(items)
    if isinstance(ckp_path,str):
        ckp = torch.load(ckp_path,map_location = lambda storage,loc:storage)
        ckp_model_dict = ckp['model']
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {remove_module_string(k):v for k,v in ckp_model_dict.items()}

    if hasattr(model,"module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)

def generate_random_seed():
    seed = hash(time.time()) % 10000
    return seed
def get_lr(optimizer):
    for param in optimizer.param_groups:
        return param['lr']


