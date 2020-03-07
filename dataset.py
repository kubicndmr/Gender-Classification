import glob 
import numpy as np
import soundfile as sf
import librosa

for item in glob.glob('dataset/*.wav'):
    [_,sr] = sf.read(item)
    [audio_data,sr] = librosa.load(item,sr=sr)
    
    if len(audio_data.shape)==2:
        print('choosing the first channel from stero sound')
        audio_data = audio_data[:,0]
        
    if not sr==8000:
        audio_data = librosa.resample(audio_data,sr,8000)
        audio_data = librosa.util.normalize(audio_data)
        sf.write(item,audio_data,8000)
