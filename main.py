import os
import csv
import copy
import librosa
import time
import torch 
import utils
import time
import pre_processing as pp 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.fftpack import fft
from scipy import signal

PP = True  #switch for using pre_processing.py

valid = 10 # Percentage of the database to use for training and validation 
test = 10

inputDim = utils.inputDim
batch_size = 8
hiddenDim = 1024
classes = 1
bias = 0.15
sampling_rate = 8000

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

class Signalset(Dataset):
    
    def __init__(self,dataList,dataLabel,path,inputDim):
        self.dataList = dataList
        self.dataLabel = dataLabel
        self.path = path
        self.inputDim = inputDim
        
    def __len__(self):
        return len(self.dataList)
    
    def __getitem__(self,idx):
        name = self.path+'/'+self.dataList[idx]
        
        features = utils.extract_features(name)
        
        if self.dataLabel[idx]=='F':
            labels = np.asarray([0]).astype(float)
        elif self.dataLabel[idx]=='M':
            labels = np.asarray([1]).astype(float)
        return features,labels


trainSet = Signalset(trainList,trainLabel,pp.TRAIN_PATH,inputDim)
validSet = Signalset(validList,validLabel,pp.VALID_PATH,inputDim)
testSet = Signalset(testList,testLabel,pp.TEST_PATH,inputDim)

trainLoader = torch.utils.data.DataLoader(dataset=trainSet,batch_size=batch_size,shuffle=True)
validLoader = torch.utils.data.DataLoader(dataset=validSet,batch_size=batch_size,shuffle=False)
testLoader = torch.utils.data.DataLoader(dataset=testSet,batch_size=batch_size,shuffle=False)
  
class Model(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_classes):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.layer3 = nn.Linear(hidden_dim,num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)

    def forward(self,x):
        a = self.relu(self.layer1(x))
        a = self.drop(a)
        a = self.relu(self.layer2(a))
        a = self.drop(a)
        a = self.layer3(a)
        return torch.sigmoid(a)

model = Model(inputDim,hiddenDim,classes)

num_params = utils.num_parameters(model)
print('Number of Parameters to be Trained: ',str(num_params))

def trainModel(model,num_epochs,loaders,learning_rate):
    
    criteria = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.5) 
    
    trnLoss = [None]*num_epochs
    valLoss = [None]*num_epochs
    val_acc= [None]*num_epochs
    
    for epoch in range(num_epochs):
        
        trn_loss = 0
        val_loss = 0
        corrects = 0
        
        scheduler.step()
        
        start = time.time()
        
        for phase in ['train','valid']:
            if phase == 'train':
                model.train()
            elif phase == 'valid':
                model.eval()
        
            for i, (fSpec,labels) in enumerate(loaders[phase]):
                f = fSpec.clone().detach().float()
                y = labels.clone().detach().float()
                                
                zero_cross = np.zeros((1,batch_size))                
                                
                optimizer.zero_grad()
                
                y_hat = model(f)
                currentCost = criteria(y_hat,y)
            
                if phase=='train':
                    currentCost.backward()
                    optimizer.step()
                    trn_loss += currentCost.item()
                
                elif phase == 'valid':
                    
                    
                    output = model(f)
                    
                    output = utils.first_decision(output,bias)
                    
                    output[output>=0.5] = 1
                    output[output<0.5] = 0
            
                    corrects += torch.sum(output== y)
                    
                    val_loss += currentCost.item()
            
           
        avg_trn_loss = trn_loss / len(trainLabel)
        avg_val_loss = val_loss / len(validLabel)
        trnLoss[epoch] = avg_trn_loss
        valLoss[epoch] = avg_val_loss
        val_acc[epoch] = 100*corrects/len(validList) 
        print('Epoch          | '+ str(epoch+1))
        print('Training Loss  | '+ str(avg_trn_loss))
        print('Validation Loss| '+ str(avg_val_loss))               
        print('Time           | '+ str(time.time()-start))
        
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(range(1,num_epochs+1),trnLoss, 'b',label='Training Loss')
        plt.plot(range(1,num_epochs+1),valLoss, 'r',label='Validation Loss')
        plt.legend(loc=1)
        fig.suptitle('Gender Classification', fontsize=24)
        plt.xlabel('epochs', fontsize=14)
        plt.ylabel('Training Loss (BCE)', fontsize=14)
        
        plt.subplot(212)
        plt.plot(range(1,num_epochs+1),val_acc, 'darkorange',label='Validation Accuracy')
        plt.xlabel('epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.savefig('plots.png')
        
dataLoader = {'train':trainLoader,'valid':validLoader}

trainModel(model=model,num_epochs=75,loaders=dataLoader,learning_rate = 1e-3)
#model.load_state_dict(torch.load('model.ckpt'))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')    
 
N = 10 
acc_hist = np.empty((1,N))
 
with torch.no_grad():
    
    for i,threshold in enumerate(np.linspace(0.1,0.9,N)):
        start = time.time()
        acc = 0
        corrects = 0
        for fSpec,labels in testLoader:
            f = fSpec.clone().detach().float()
            y = labels.clone().detach().float()
                
            output = model(f)
            
            output = utils.first_decision(output,bias)
            output[output>=threshold] = 1
            output[output<threshold] = 0
            
            corrects += torch.sum(output== y)
        
        acc = 100*corrects/len(testList)
        print(acc)
        acc_hist[0][i] = acc
        print(time.time()-start)
        
    print(acc_hist)
    print('Accuracy of the network on the {} test speech signals: {}%'.format(len(testList),np.max(acc_hist)))
