import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,feature_size):
        self.feature_size = feature_size
        super(LSTM,self).__init__()
        self.bn1 = nn.ReLU()
        self.lstm = nn.LSTM(  
            input_size=self.feature_size, #feature_size在UCI Dataset中为561维度
            hidden_size=32,    
            num_layers=5,       
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)

        )
        self.fc = nn.Linear(32,6)

    def forward(self,x):
        x = self.bn1(x)
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc(r_out[:, -1, :])
        return out

        