import os
import librosa

DATABASE_PATH = 'Lombard_Speech_Database_German/'
TARGET_PATH = 'dataset/'

if not os.path.exists(TARGET_PATH):
    os.makedirs(TARGET_PATH)

#Copy all .wav files in subdirs to a single dir
for dirs, subdirs, files in os.walk(DATABASE_PATH):
    for name in files:
        shutil.copy(dirs + '/' + name, TARGET_PATH)

#Resample to 16k
for wav in os.listdir(TARGET_PATH):
    resampled,_ = librosa.load(TARGET_PATH+wav,sr=16000,duration=2.0)
    librosa.output.write_wav(TARGET_PATH+wav,resampled,16000)


    
    
    
