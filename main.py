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
inputDim = 512
hiddenDim = 1024
classes = 1

if PP:
    pp.fileOrder(valid,test)
    pp.resampleDatabase(sampling_rate)

# Device Configuration
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
        return x[offset:offset+self.inputDim*2]
    
    def __getitem__(self,idx):
        wav,sr = librosa.load(self.path+'/'+self.dataList[idx],sr=16000)
        wav = TrainingDataset.window(self,wav,sr)
              
        f = np.abs(fft(wav))
        f = f[self.inputDim:]
        f = f / np.linalg.norm(f)
                
        if self.dataLabel[idx]=='f':
            l = np.asarray([0]).astype(float)
        elif self.dataLabel[idx]=='m':
            l = np.asarray([1]).astype(float)
        return f,l


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
        dur = int(len(x)/self.sampling_rate)
        offset = self.randomArray[idx]*dur
        return x[offset:offset+self.inputDim*2]
    
    def __getitem__(self,idx):
        wav,sr = librosa.load(self.path+'/'+self.dataList[idx],sr=self.sampling_rate)
        wav = TestingDataset.window(self,wav,idx)
        
        f = np.abs(fft(wav))
        f = f[self.inputDim:]
        f = f / np.linalg.norm(f)
        
        if self.dataLabel[idx]=='f':
            l = np.asarray([0]).astype(float)
        elif self.dataLabel[idx]=='m':
            l = np.asarray([1]).astype(float)
        return f,l

np.random.seed(1)
randomArrayVal = np.random.randint(0,sampling_rate-inputDim-1,size=len(validList))
np.random.seed(2)
randomArrayTe = np.random.randint(0,sampling_rate-inputDim-1,size=len(testList))

testSet = TestingDataset(testList,testLabel,pp.TEST_PATH,inputDim,randomArrayTe,sampling_rate)
validSet = TestingDataset(validList,validLabel,pp.VALID_PATH,inputDim,randomArrayVal,sampling_rate)

trainLoader = torch.utils.data.DataLoader(dataset=trainSet,batch_size=4,shuffle=True)
validLoader = torch.utils.data.DataLoader(dataset=validSet,batch_size=4,shuffle=False)
testLoader = torch.utils.data.DataLoader(dataset=testSet,batch_size=8,shuffle=False)
  
class Model(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_classes):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.layer3 = nn.Linear(hidden_dim,num_classes)
        self.drop = nn.Dropout(p=0.2)

    def forward(self,x):
        a = self.layer1(x)
        a = self.drop(a)
        a = self.layer2(a)
        a = self.drop(a)
        a = self.layer3(a)
        return torch.sigmoid(a)

model = Model(inputDim,hiddenDim,classes)


def trainModel(model,num_epochs,loaders,learning_rate):
    
    criteria = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1) 
    trnLoss = [None]*num_epochs
    valLoss = [None]*num_epochs
    lr_hist = [None]*num_epochs
    
    for epoch in range(num_epochs):
        
        trn_loss = 0
        val_loss = 0
        scheduler.step()
        
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
                    val_loss += currentCost.item()
                    
        #early stopping
        # en iyi networku kaydet
        avg_trn_loss = trn_loss / len(trainLabel)
        avg_val_loss = val_loss / len(validLabel)
        trnLoss[epoch] = avg_trn_loss
        valLoss[epoch] = avg_val_loss
        lr_hist[epoch] = scheduler.get_lr() 
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

trainModel(model=model,num_epochs=100,loaders=dataLoader,learning_rate = 1e-3)
#model.load_state_dict(torch.load('model.ckpt'))
 
N = 10 
acc_hist = np.empty((1,N))
 
with torch.no_grad():
    
    for i,threshold in enumerate(np.linspace(0.1,0.9,N)):
        
        acc = 0
        corrects = 0
        for fSpec,labels in testLoader:
            f = fSpec.clone().detach().float()
            y = labels.clone().detach().float()
        
            output = model(f)
    
            output[output>=threshold] = 1
            output[output<threshold] = 0
        
            corrects += torch.sum(output== y)
        
        acc = 100 * corrects / len(testList)
        acc_hist[0][i] = acc
        
    print(acc_hist)
    print('Accuracy of the network on the {} test speech signals: {}%'.format(len(testList),np.max(acc_hist)))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')    
        
        
        
        
    
