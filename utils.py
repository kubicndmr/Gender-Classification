import numpy as np
import os
import csv
from scipy.io import wavfile
import scipy.signal
import librosa
from scipy.fftpack import fft
from scipy import signal
import torch

inputDim = 512

N = 10
Wn= 0.4
b,a = signal.butter(N,Wn,btype='low',output='ba') 

def generate_labels(path):
    wavs = os.listdir(path)
    labels = list()
    
    for item in wavs:
        if item.startswith('F'):
            labels.append((item,'F'))
        elif item.startswith('M'):
            labels.append((item,'M'))
        
    with open(path+"labels.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(labels)
        
def load_labels(path):
    List,Label = list(), list()
    with open(path+'labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            List.append(row[0])
            Label.append(row[1])
    return List,Label

def extract_features(name):
    sr,wav = wavfile.read(name)
    filtered = signal.lfilter(b,a,wav)
    freq = np.abs(fft(filtered))
    semi = freq[len(freq)//2:]
    trimmed, index = librosa.effects.trim(semi,26,frame_length=512, hop_length=128)
    resampled = signal.resample(trimmed,inputDim)
    return resampled / np.max(resampled)

def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def zero_crossing(x):
    return len(np.where(np.diff(np.sign(x)))[0])

def first_decision(output,bias):
    output = output.detach().numpy()
    b = bias
    for x in output:
        if 0.5<x<0.85 and zero_crossing(x)>7250:
            x += b
        if 0.15<x<0.5 and zero_crossing(x)<7250:
            x -= b
    return torch.from_numpy(output)
    
