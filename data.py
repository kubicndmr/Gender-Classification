from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import torch
import glob
import random

valid_ratio = .25

class SpeechDataset(Dataset):
    def __init__(self,mode,win_len,data_path,valid_ratio,transforms):
        self.mode = mode
        self.win_len = win_len
        self.transforms = transforms
        self.path = data_path
        self.get_data(data_path,valid_ratio)
        
    def __getitem__(self,index):
        data_path = self.data[index]
        data = data_path.split('\\')[-1]
        
        #choose augmentation and load data
        if self.mode=='train':
            aug = index%len(self.transforms)  
            tv = self.transforms[aug]
        elif self.mode=='val':
            tv = 'clean'
        elif self.mode=='test':
            tv = 'clean'
        
        if tv=='clean':
            speech,fs = sf.read(self.path+data)
        elif tv=='revb':
            speech,fs = sf.read('revb/'+data)
        elif tv=='noise':
            speech,fs = sf.read('noise/'+data)
        elif tv=='reverse':
            speech,fs = sf.read(self.path+data)
            speech = speech[::-1].copy()
        else:
            print('Augmentation type not understood!')
            
                
        # take a random window
        start = int(np.random.randint(0,len(speech)-self.win_len,1))
        stop = int(start + self.win_len)
        speech = speech[start:stop]
        speech = torch.from_numpy(speech)
        
        # convert label to torch tensor
        label = self.read_label(data)
        label = torch.tensor(label)
        
        return (speech,label)
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self,path,vr):
        Data = glob.glob(path+'*.wav')
        
        random.shuffle(Data)
        
        vr = int(np.floor(len(Data)*vr))  
        if self.mode == 'train':
            self.data = Data[vr:]
            self.data = sorted(self.data*len(self.transforms))
        elif self.mode == 'val':
            self.data = Data[:vr]
        elif self.mode =='test':
            self.data = Data
            
    def read_label(self,data_name):
        
        if data_name[0] =='M':
            label = [1,0]
        elif data_name[0] =='F':
            label = [0,1]
        else:
            print('Label type not understood')
            label = -1
        return label
    
    def pos_weight(self): #Useful for imbalanced datasets
        positives = np.empty((1,2))
        
        for [_,l] in self.data:
            label = self.read_label(l) 
            positives += label
        
        negatives = np.array([len(self.data)-positives[0,0],len(self.data)-positives[0,1]])
        return negatives/positives
    
def get_train_dataset():
    return SpeechDataset('train',8000,'dataset/',valid_ratio,['clean','revb','noise'])

def get_validation_dataset():
    return SpeechDataset('val',8000,'dataset/',valid_ratio,['clean'])

def get_test_dataset():
    return SpeechDataset('test',8000,'testset/',valid_ratio,['clean'])

