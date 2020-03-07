# Gender-Classification with PyTorch
Gender Classification of Speech Signals

Notes
--------
This project aims to classify gender of the speaker using neural network model. 

Usage
--------

  First, required dependencies should be installed:

environment.yml => dependencies

  Second step is preparing data:  

I've used sub set of the VCTK clean speech databases consist of equal number of female and male English speakers. Database can be found here:  https://datashare.is.ed.ac.uk/handle/10283/1942

Dataset should be under 'dataset/' folder in the same dir. If not, paths in scripts should be corrected accordingly. In order to have larger datasets, I've used noisy and reverberant versions of the clean speech dataset. You can use the following scripts to prepare required datas.

dataset.py => Resample dataset files to 8 kHz

testset.py => Picks randomly 100 samples from dataset and moves to 'testset/'. 

bg_noise_generator.py => Download random background noises from YouTube using Google Audioset. For more info: https://research.google.com/audioset/index.html

noise.py => This file copies samples from dataset, adds noise and saves under 'noise/'

revb.mat => This file copies samples from dataset, adds reverberation and saves under 'revb/'

  Third step is training Network:

model/GenderClassifierModel.py => Defines the model

data.py => Controls flow of data 

stopping.py => Early Stoppping callback 

trainer.py => Controls the training steps

train.py => Defines model and trains the network for given parameters

Running train.py will train the network and save model checkpoints. 

  Last step is the testing:
 
test.py => Tests every sample in testset and saves estimations with confidence levels into a csv file.

With provided pre-trained model, I get approximately %97 percent accuracy.  
