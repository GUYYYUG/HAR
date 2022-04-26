
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torchvision.transforms import ToPILImage
import torchvision
import torch.utils.model_zoo as model_zoo
import numpy as np
import os
import sys
import time
import random
import string
import yaml
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
# ==== project
from model_zoo import LSTM
from lab_dataset import motionset
from utils import save_model,generate_random_seed,get_lr


class Trainer():
    def __init__(self,config_file):
        self.set_seed()
        params = self.load_config_file(config_file)
        random_string = np.random.choice(list(string.digits+string.ascii_letters),size=(10,))
        random_string = "".join(random_string)
        params.update({"seed":self.seed,
                       "random_string":random_string})
        self.update_attrs(params)
        self.make_model()
        self.make_dataset()
        self.run()

    def load_config_file(self,config_file):
        return yaml.safe_load(open(config_file,"r"))
    
    def set_seed(self):
        seed = generate_random_seed()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.seed = seed

    def update_attrs(self,kwargs):
        
        self.__dict__.update(kwargs)
    
    def make_dataset(self):
        trainset = motionset("./train.pkl",'train',self.feature_size)
        valset = motionset("./test.pkl",'valid',self.feature_size)
        testset = motionset("./test.pkl",'test',self.feature_size)
        print("len of train: ",len(trainset))
        print("len of val: ",len(valset))
        print("len of test: ",len(testset))
        self.trainloader = DataLoader(dataset=trainset,batch_size=self.batch_size,shuffle=True,num_workers=0,pin_memory=False)
        self.testloader = DataLoader(dataset=testset,batch_size=self.batch_size,shuffle=False,num_workers=0,pin_memory=False)
        self.valloader = DataLoader(dataset=valset,batch_size=self.batch_size,shuffle=False,num_workers=0,pin_memory=False)
    
    def make_model(self,):
        
        self.model = LSTM(self.feature_size)
        # print(self.num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=9e-4)
        
        # self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0015)
        print("gpus num",len(self.gpus))
        if len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model,device_ids=self.gpus)
        self.main_dev = torch.device(f"cuda:{self.gpus[0]}")
        self.model.to(self.main_dev)
        self.reg_criterion = nn.CrossEntropyLoss().to(self.main_dev)

    def run(self,):
        start_epoch = 0 if "start_epoch" not in self.__dict__ else self.start_epoch
        self.train_losses = []
        self.train_counter = []
        self.train_acc = []
        self.train_loss = []
        self.valid_acc = []
        self.valid_loss = []
        for epoch in range(start_epoch,start_epoch+self.num_epochs):
            self.train(epoch)
            self.evaluate(epoch,'test',self.testloader)
        save_model(self.model,self.optimizer,epoch,os.path.join(self.ckp_dir,"best.pth.tar"))
        self.evaluate(self.num_epochs,"test",self.testloader)
        fig = plt.figure()
        # print(self.train_counter)
        plt.plot(self.train_counter, self.train_losses, color='blue')
        plt.legend(['Train Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
    
    def train(self,epoch):
        self.model.train()
        pbar = tqdm(self.trainloader)
        running_loss = 0.0
        running_loss = 0.0
        correct = 0
        processed = 0
        
        for iterx, (data, target) in enumerate(pbar):
            data = data.type(torch.FloatTensor)
            # print(data.shape)
            # print(target)
            data = data.reshape(-1,1,self.feature_size)
            data, target = data.to(self.main_dev), target.to(self.main_dev)
        
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            # print(y_pred)
            # if iterx == 1:
            #     break
            loss = self.reg_criterion(y_pred, target.long())
            running_loss += loss.item()
            self.train_loss.append(loss)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            pred = y_pred.argmax(dim=1, keepdim=True) 
            # print(pred) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            self.train_acc.append(100*correct/processed)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={iterx} le={get_lr(self.optimizer)} Accuracy={100*correct/processed:0.2f}')
    def evaluate(self,epoch,context,dataloader):
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.testloader:
                data = data.type(torch.FloatTensor)
                data = data.reshape(-1,1,self.feature_size)
                data, target = data.to(self.main_dev), target.to(self.main_dev)
                
                output = self.model(data)
                # print(output)
                test_loss += criterion(output, target.long()).item()  
                pred = output.argmax(dim=1, keepdim=True)  
                # print(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testloader.dataset)
        self.valid_loss.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.testloader.dataset),
            100. * correct / len(self.testloader.dataset)))
        
        self.valid_acc.append(100. * correct / len(self.testloader.dataset))
 

if __name__ == '__main__':
    t = Trainer("./train.yaml")