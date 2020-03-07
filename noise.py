import numpy as np 
import glob
import os
import soundfile as sf
import random 
import sys
import matplotlib.pyplot as plt

if not os.path.exists('noise/'):
    os.mkdir('noise/')

bg_noise_items = os.listdir('BG_noise')

for speech_item in glob.glob('dataset/*.wav'):
    item_checker = speech_item.split('\\')[1]
    if not item_checker in os.listdir('noise/'):
        [speech_data,sr] = sf.read(speech_item)
        Es = np.sum(speech_data**2) # Signal energy
        
        #choose random background noise
        bg_noise = random.choice(bg_noise_items)
        [bg_noise,sr_n] = sf.read('BG_noise/'+bg_noise)
        
        #check samples rates
        if not sr_n==sr:
            print('sampling rates are not correct!')
        
        # take window or zero pad at the end to make both same length
        if len(bg_noise)>len(speech_data):
            start = int(np.random.randint(low=0,high=(len(bg_noise)-len(speech_data)),size=1))
            stop = start + len(speech_data)
            bg_noise = bg_noise[start:stop]
        else:
            bg_noise = np.pad(bg_noise,(0,len(speech_data)-len(bg_noise)),'constant',constant_values=(0))
            
            
        En = np.sum(bg_noise**2) # Noise energy
        
        #set 10dB difference in energies
        while Es*0.05<En:
            bg_noise /=2
            En = np.sum(bg_noise**2) # Noise energy
        
        #save
        print(speech_item)
        file_name = 'noise/'+speech_item.split('\\')[-1] #this split is for windows, use '/' for linux
        speech_w_bg = speech_data + bg_noise
        sf.write(file_name,speech_w_bg,sr)
    
    
