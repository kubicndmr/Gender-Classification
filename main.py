import os
import csv
import copy
import librosa
import torch 
import utils
import time
import pre_processing as pp 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.fftpack import fft

PP = False #switch for using pre_processing.py

valid = 10 # Percentage of the database to use for training and validation 
test = 10
sampling_rate = 16000
inputDim = 256
hiddenDim = 1024
classes = 1

if PP:
    pp.fileOrder(valid,test)
    pp.resampleDatabase(sampling_rate)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Process Dataset
trainList,trainLabel = list(),list()
validList,validLabel = list(),list()
testList,testLabel = list(),list()

utils.generate_labels(pp.TRAIN_PATH)
utils.generate_labels(pp.VALID_PATH)
utils.generate_labels(pp.TEST_PATH)

trainList,trainLabel = utils.load_labels(pp.TRAIN_PATH)
validList,validLabel = utils.load_labels(pp.VALID_PATH)
testList,testLabel = utils.load_labels(pp.TEST_PATH)

class TrainingDataset(Dataset):
    
    def __init__(self,dataList,dataLabel,path,inputDim):
        self.dataList = dataList
        self.dataLabel = dataLabel
        self.path = path
        self.inputDim = inputDim
        
    def __len__(self):
        return len(self.dataList)
    
    def window(self,x,sr):
        offset = np.random.randint(0,sr-inputDim-1)
        return x[offset:offset+self.inputDim]
    
    def __getitem__(self,idx):
        wav,sr = librosa.load(self.path+'/'+self.dataList[idx],sr=16000,duration=2.0)
        wav = TrainingDataset.window(self,wav,sr)
        f = np.abs(fft(wav))
        if self.dataLabel[idx]=='f':
            l = np.asarray([0]).astype(float)
        elif self.dataLabel[idx]=='m':
            l = np.asarray([1]).astype(float)
        return [f,l]


trainSet = TrainingDataset(trainList,trainLabel,pp.TRAIN_PATH,inputDim)


class TestingDataset(Dataset):
    
    def __init__(self,dataList,dataLabel,path,inputDim,randomArray,sampling_rate):
        self.dataList = dataList
        self.dataLabel = dataLabel
        self.path = path
        self.inputDim = inputDim
        self.randomArray = randomArray
        self.sampling_rate = sampling_rate
        
    def __len__(self):
        return len(self.dataList)
    
    def window(self,x,idx):
        offset = self.randomArray[idx]
        return x[offset:offset+self.inputDim]
    
    def __getitem__(self,idx):
        wav,sr = librosa.load(self.path+'/'+self.dataList[idx],sr=16000,duration=2.0)
        wav = TestingDataset.window(self,wav,idx)
        f = np.abs(fft(wav))
        if self.dataLabel[idx]=='f':
            l = np.asarray([0]).astype(float)
        elif self.dataLabel[idx]=='m':
            l = np.asarray([1]).astype(float)
        return [f,l]


np.random.seed(101)
randomArrayVal = np.random.randint(0,sampling_rate-inputDim-1,size=len(validList))
np.random.seed(102)
randomArrayTe = np.random.randint(0,sampling_rate-inputDim-1,size=len(testList))

validSet = TestingDataset(validList,validLabel,pp.VALID_PATH,inputDim,randomArrayVal,sampling_rate)
testSet = TestingDataset(testList,testLabel,pp.TEST_PATH,inputDim,randomArrayTe,sampling_rate)

trainLoader = torch.utils.data.DataLoader(dataset=trainSet,batch_size=4,shuffle=True)
validLoader = torch.utils.data.DataLoader(dataset=validSet,batch_size=4,shuffle=False)
testLoader = torch.utils.data.DataLoader(dataset=testSet,batch_size=8,shuffle=False)


class Model(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_classes):
        super(Model, self).__init__()
        self.layer1 = nn.Conv1d(in_dim,hidden_dim,kernel_size=11,padding=5)
        self.layer2 = nn.Conv1d(hidden_dim,hidden_dim,kernel_size=11,padding=5)
        self.layer3 = nn.Linear(hidden_dim,num_classes)
        self.activation = nn.Softmax(dim=0)
        
    def forward(self,x):
        x = x.unsqueeze(-1)
        a = self.layer1(x)
        a = self.activation(a)
        a = self.layer2(a)
        a = self.activation(a)
        a = a.squeeze()
        a = self.layer3(a)
        return a

model = Model(inputDim,hiddenDim,classes)


def trainModel(model,num_epochs,loaders,learning_rate):
    
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    trnLoss = [None]*num_epochs
    valLoss = [None]*num_epochs
    
    for epoch in range(num_epochs):
        
        trn_loss = 0
        corrects = 0
        
        for phase in ['train','valid']:
            if phase == 'train':
                model.train()
            elif phase == 'valid':
                model.eval()
        
            for i, (fSpec,labels) in enumerate(loaders[phase]):
                f = fSpec.clone().detach().float()
                y = labels.clone().detach().float()
            
                optimizer.zero_grad()
                
                y_hat = model(f)
                currentCost = criteria(y_hat,y)
            
                if phase=='train':
                    currentCost.backward()
                    optimizer.step()
                    trn_loss += currentCost.item()
                
                elif phase == 'valid':
                    output = model(f)
                    output[output>=0.5] = 1
                    output[output<0.5] = 0
                    corrects += torch.sum(output== y)
                        
        avg_trn_loss = trn_loss / len(trainLabel)
        avg_val_loss = corrects.numpy() / len(validLabel)
        trnLoss[epoch] = avg_trn_loss
        valLoss[epoch] = avg_val_loss
        print('Epoch          | '+ str(epoch+1))
        print('Training Loss  | '+ str(avg_trn_loss))
        print('Validation Loss| ' + str(avg_val_loss))
    
    fig = plt.figure()
    plt.plot(range(num_epochs),trnLoss, 'b',label='Training Loss')
    plt.plot(range(num_epochs),valLoss, 'r',label='Validation Loss')
    plt.legend(loc=1)
    fig.suptitle('Gender Classification', fontsize=24)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('Training Loss (MSE)', fontsize=14)
    plt.savefig('LossFunc.png')


dataLoader = {'train':trainLoader,'valid':validLoader}

trainModel(model=model,num_epochs=100,loaders=dataLoader,learning_rate = 1e-4)
        
with torch.no_grad():
    corrects = 0
    
    for fSpec,labels in testLoader:
        f = fSpec.clone().detach().float()
        y = labels.clone().detach().float()
        
        output = model(f)
        output[output>=0.5] = 1
        output[output<0.5] = 0
        
        corrects += torch.sum(output== y)

    print('Accuracy of the network on the 23 test speech signals: {}%'.format(100 * corrects / len(testList)))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')    
        
        
        
        
    
