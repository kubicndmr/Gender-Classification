from data import get_train_dataset, get_validation_dataset
import model.GenderClassifierModel as gcm
from trainer import Trainer
from stopping import EarlyStoppingCB
import matplotlib.pyplot as plt
import numpy as np
import torch as t

epoch = 50

trainSet = get_train_dataset()
validSet = get_validation_dataset()

print("Training set size (augmented): ",trainSet.__len__(),"samples")
print("Validation set size: ",validSet.__len__(),"samples")

trainLoader = t.utils.data.DataLoader(dataset=trainSet,batch_size=128,shuffle=True)
validLoader = t.utils.data.DataLoader(dataset=validSet,batch_size=128,shuffle=True)

# set up your model
model = gcm.GCM()

# set up loss 
criteria = t.nn.MSELoss()

# set up optimizer (see t.optim); 
optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

#set up scheduler
scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

#set up early stopper
early_stop = EarlyStoppingCB(patience=5,verbose=True)

# initialize the early stopping callback implemented in stopping.py and create a object of type Trainer
trainer = Trainer(model,criteria,optimizer,scheduler,trainLoader,validLoader,early_stop)

print('Number of trainable parameters: ',sum(p.numel() for p in model.parameters() if p.requires_grad))

res = trainer.fit(epoch)
trainer.restore_checkpoint(epoch-1)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch-1))

## plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('loss.png',format='png')
plt.show()
