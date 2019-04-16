import os
import csv
import librosa
import torch 
import utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.fftpack import fft

DATA_PATH = 'dataset/'
TEST_PATH = 'testset/'

inputDim = 256
hiddenDim = 512
classes = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Process Dataset
dataList,dataLabel = list(),list()
testList,testLabel = list(),list()

utils.generate_labels(DATA_PATH)
utils.generate_labels(TEST_PATH)

dataList,dataLabel = utils.load_labels(DATA_PATH)
testList,testLabel = utils.load_labels(TEST_PATH)


class SpeechDataset(Dataset):
    
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
        wav = SpeechDataset.window(self,wav,sr)
        f = np.log(np.abs(fft(wav)))
        if self.dataLabel[idx]=='f':
            l = np.asarray([0]).astype(float)
        elif self.dataLabel[idx]=='m':
            l = np.asarray([1]).astype(float)
        return [f,l]
    

trainSet = SpeechDataset(dataList,dataLabel,DATA_PATH,inputDim)
testSet = SpeechDataset(testList,testLabel,TEST_PATH,inputDim)



trainLoader = torch.utils.data.DataLoader(dataset=trainSet,batch_size=4,shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testSet,batch_size=8,shuffle=True)


class Model(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_classes):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.layer3 = nn.Linear(hidden_dim,num_classes)
        
    def forward(self,x):
        a = self.layer1(x)
        a = self.layer2(a)
        a = self.layer3(a)
        return torch.sigmoid(a)
    

model = Model(inputDim,hiddenDim,classes)


def trainModel(model,numIter,loader,learning_rate):
    cost = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    loss = [None]*numIter
    
    for epoch in range(numIter):
        average_cost = 0
        for i, (fSpec,labels) in enumerate(loader):
            f = torch.tensor(fSpec,dtype=torch.float,device=device)
            y = torch.tensor(labels,dtype=torch.float,device=device)
            
            y_hat = model(f)
            currentCost = cost(y_hat,y)
            
            optimizer.zero_grad()
            currentCost.backward()
            optimizer.step()
            
            average_cost += currentCost.item()
            
        average_cost = average_cost / len(dataLabel)    
        loss[epoch] = average_cost
        print('Epoch| '+ str(epoch+1))
        print('Loss | '+ str(loss[epoch]))
    
    fig = plt.figure()
    plt.plot(range(numIter),loss)
    fig.suptitle('Gender Classification', fontsize=24)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('Loss (Cross Entropy)', fontsize=14)
    plt.savefig('LossFunc.png')

trainModel(model,150,trainLoader,0.0005)
        
with torch.no_grad():
    correct = 0
    total = len(testLabel)
    for fSpec,labels in testLoader:
        label = labels.to(device)
        output = model(fSpec)
        estimate = np.empty([1,len(label)])
        for i in range(len(label)):
            if output[i] >= 0.5:
                estimate[0][i] = 1
            elif output[i] < 0.5:
                estimate[0][i] = 0
            print('Ground Truth| '+str(label[i].item())+' Estimated| '+str(estimate[0][i]))
                
            correct += (estimate[0][i] == label[i].item())*1

    print('Accuracy of the network on the 23 test speech signals: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')    
        
        
        
        
    
