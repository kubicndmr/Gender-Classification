import os
import librosa
import shutil
import random

DATABASE_PATH = 'Lombard_Speech_Database_German/'
TRAIN_PATH = 'dataset/'
VALID_PATH = 'validset/'
TEST_PATH = 'testset/'

def fileOrder(valid,test):

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    
    if not os.path.exists(VALID_PATH):
        os.makedirs(VALID_PATH)
        
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)    

    temp_paths = list()
    
    #Copy all .wav files in subdirs to a single dir
    for dirs, subdirs, files in os.walk(DATABASE_PATH):
        for name in files:
            shutil.copy(dirs + '/' + name, TRAIN_PATH)
            temp_paths.append(name)
    
    numData = len(temp_paths)
    numValid = numData // valid
    numTest = numData // test
    
    randomSamples = random.sample(temp_paths,k=numValid+numTest) 
    
    for name in randomSamples[:numValid]:
        shutil.move(TRAIN_PATH+name,VALID_PATH)
        
    for name in randomSamples[numTest:]:
        shutil.move(TRAIN_PATH+name,TEST_PATH)        
    
    

def resampleDatabase(sampling_rate):
    
    paths = [TRAIN_PATH,VALID_PATH,TEST_PATH]
    
    #Resample to 16k
    for TARGET_PATH in paths:
        for wav in os.listdir(TARGET_PATH):
            resampled,_ = librosa.load(TARGET_PATH+wav,sr=sampling_rate)
            librosa.output.write_wav(TARGET_PATH+wav,resampled,sampling_rate)


    
    
    
