from __future__ import unicode_literals
import youtube_dl
import numpy as np
import pandas as pd
import random 
import soundfile as sf
import librosa
import glob
import os

def getBGNoise(duration_limit):
    
    # Protection from infinite loops
    iter_limit = 45
    
    # Create dir
    if not os.path.exists('BG_noise'):
        os.mkdir('BG_noise')
    
    # Remove remaning wav files for ease of use 
    for r in glob.glob('./*.wav'):
        print('Removing ',r)
        os.remove(r)
    
    # Read noise csv
    dataList = pd.read_csv('noises.csv').values.tolist()
    
    duration = 0
    
    while duration < duration_limit: # in seconds
        
        noiseFlag = 1
        
        # For noise types see: https://github.com/audioset/ontology/blob/master/ontology.json
        noiseType = ["/m/01sm1g","/m/06w87","/m/078jl","/m/09ld4","/m/015p6","/m/03k3r","/m/04gxbd","/m/04zmvq","/t/dd00067","/t/dd00130","/m/07r67yg","/g/11b630rrvh","/m/012xff","/m/01lsmm","/m/0242l","/m/01m4t","/m/0dv5r","/m/01b82r","/m/0_1c"]
        avoidSpeech = set(["/m/0dgw9r","/m/09l8g","/m/09x0r","/m/05zppz","/m/02zsn","/m/0ytgt","/m/01h8n0","/m/02qldy","/m/0261r1","/m/0brhx","/m/07p6fty", "/m/07sr1lc","/m/03qc9zr","/m/01j3sz","/m/015lz1"])
        
        noiseType = set(random.choices(noiseType,k=3))
        
        iter_count = 0 
        
        print('Searching youtube for appropiate sound')
        while noiseFlag and iter_count<iter_limit: 
            data = random.choice(dataList)
            data = data[0].split(',')
            
            youtubeID = data[0]
            start = int(data[1].split('.')[0])
            stop = int(data[2].split('.')[0])
            
            labels = set(data[3:])
            
            if len(noiseType.intersection(labels))>0 and len(avoidSpeech.intersection(labels))==0: 
                noiseFlag=0
            
            iter_count += 1
            
        ydl_opts = {
            'format': 'best/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }]}
        
        try:
            # Download audio
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                aud = ydl.download(["http://www.youtube.com/watch?v="+youtubeID])       
            
            # Window, save and clean
            noise_file = glob.glob('./*.wav')
            noise_file = noise_file[0].split('//')[-1]
            [audio_data,sr] = sf.read(noise_file)
            start *= sr
            stop *= sr
            audio_data = audio_data[start:stop]
            idx = len(os.listdir('BG_noise'))
            sf.write('BG_noise/BG_noise_'+str(idx+1)+'.wav',audio_data,sr)
            os.remove(noise_file)
                
            duration += ((stop-start)/sr)
            print("Current duration of added background noise is %d seconds" % duration)
            
        except:
            print('Passing this sample')
    
    # Process downloaded files in noise dir
    print('Processing downloaded files')
    for noise_sample in glob.glob('./BG_noise/*wav'):
        [_,sr] = sf.read(noise_sample)
        if sr != 8000:
            [audio_data,sr] = librosa.load(noise_sample,sr=sr)
            print(noise_sample,str(sr),len(audio_data.shape))
            if len(audio_data.shape)==2:
                print('choosing the first channel from stero sound')
                audio_data = audio_data[:,0]
            audio_data = librosa.resample(audio_data,sr,8000)
            audio_data = librosa.util.normalize(audio_data)
            sf.write(noise_sample,audio_data,8000)


if __name__ == "__main__":
    
    # Find total duration of training dataset
    dur = 0
    for audio_file in glob.glob('dataset/*.wav'):
        [audio_data,_] = sf.read(audio_file)
        dur += len(audio_data)

    dur = dur//16000    
    print("Training dataset duration is %d s" % dur)

    # Find existing length of noise dataset
    ndur = 0
    if os.path.exists('BG_noise/'):
        for noise_file in glob.glob('BG_noise/*.wav'):
            [noise_data,sr] = sf.read(noise_file)
            ndur += (len(noise_data)/sr)

    ndur = int(ndur)
    print("Existing noise dataset duration is %d s" % ndur)

    genNoiseDur = int((dur - ndur)*.25)

    if genNoiseDur > 0:
        print("%d s of new noise will be generated" % genNoiseDur)

        # Generate noise dataset 
        getBGNoise(genNoiseDur)
        print('Noise generation is done!!!')
