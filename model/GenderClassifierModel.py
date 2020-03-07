import numpy as np
import data as dataLoader
import torch
from torch import nn   

class GCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in1 = nn.Conv1d(1,10,kernel_size=1,stride=1,padding=1)
        self.conv_in2 = nn.Conv1d(10,10,kernel_size=3,stride=1)
        self.maxpool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2000,256)
        nn.init.kaiming_normal_(self.fc1.weight,nonlinearity='relu')
        self.fc2 = nn.Linear(128,128)
        nn.init.kaiming_normal_(self.fc2.weight,nonlinearity='relu')
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.conv_out1 = nn.Conv1d(10,1,kernel_size=1,stride=2)
        self.conv_out2 = nn.Conv1d(1,1,kernel_size=1,stride=2)
        self.fc3 = nn.Linear(64,2)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv_in1(x)
        x = self.conv_in2(x)
        x = self.maxpool(x)
        x = self.relu(self.fc1(x))
        x = self.conv_out1(x)
        x = self.relu(self.fc2(x))
        x = self.conv_out2(x)
        x = self.fc3(x)
        return self.softmax(x).squeeze()
