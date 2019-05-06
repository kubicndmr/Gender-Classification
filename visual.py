import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def window(x,sr):
    offset = np.random.randint(0,sr-255)
    return x[offset:offset+256]
    
wav,sr = librosa.load('female.wav',sr=16000,duration=2.0)

wav = window(wav,sr)

t=np.arange(256)

freq = np.fft.fftfreq(t.shape[-1])
f = np.log(np.abs(fft(wav)))
plt.figure()
plt.plot(freq,f)
#plt.show()

wav,sr = librosa.load('male.wav',sr=16000,duration=2.0)

wav = window(wav,sr)

f = np.log(np.abs(fft(wav)))
plt.figure()
plt.plot(freq,f)
plt.show()






