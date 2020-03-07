import model.GenderClassifierModel as gcm
from data import get_test_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import csv
import os

# load model
model = gcm.GCM()

# restore last checkpoint
last_checkpoint = len(os.listdir('checkpoints'))-1
if last_checkpoint < 100:
    last_checkpoint = '0'+str(last_checkpoint)
model.load_state_dict(t.load(f'checkpoints/checkpoint_{last_checkpoint}.ckp')['state_dict'])
model.eval()
model.double()

# get dataset
testSet = get_test_dataset()
testLoader = t.utils.data.DataLoader(dataset=testSet,batch_size=1,shuffle=False)

results = []
tp = 0

# test
with t.no_grad():
    for item in testLoader:
        x = item[0].double()
        y = item[1].numpy()
    
        y_hat = model.forward(x)
        y_hat = y_hat.numpy()
        
        #print(np.allclose(y_hat,y,atol=0.15,rtol=0.15),y_hat,y)
        results.append([y_hat,y])
        if np.allclose(y_hat,y,atol=0.2,rtol=0.2):
            tp +=1
print(tp)
print(len(os.listdir('testset/')))
acc = tp/len(os.listdir('testset/'))*100
print(f'Accuracy in test set is %{acc} percent')

#save results

with open('results.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerows(results)
    
