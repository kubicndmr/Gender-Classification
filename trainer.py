import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
import os

class Trainer:
    
    def __init__(self,               
                 model,                # Model to be trained.
                 crit,                 # Loss function
                 optim = None,         # Optimiser
                 shed = None,          # Scheduler
                 train_dl = None,      # Training data set
                 val_test_dl = None,   # Validation (or test) data set
                 early_stopping_cb = None): # The stopping criterion.
        self._model = model
        self._crit = crit
        self._optim = optim
        self._shed = shed
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._early_stopping_cb = early_stopping_cb
        self.device = t.device("cpu")
        
        if t.cuda.is_available():
            self._model = model.cuda()
            self._crit = crit.cuda()
            self.device = t.device("cuda")
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(3, 1024, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        self._optim.zero_grad()
        y_hat = self._model(x)
        e = self._crit(y_hat,y)
        e.backward()
        self._optim.step()
        return e
        
    def val_test_step(self, x, y):
        with t.no_grad():
            y_hat = self._model(x)
            e = self._crit(y_hat,y)
            y_hat = t.round(y_hat)
            return (e,y_hat)
    
    def train_epoch(self):
        self._model.train()
                
        epoch_loss = 0
        
        for i, (x_,y_) in enumerate(self._train_dl):
            x = x_.clone().detach().float().to(self.device)
            y = y_.clone().detach().float().to(self.device)
            epoch_loss += self.train_step(x,y)
            
        return (epoch_loss / i).item()
    
    def val_test(self):
        self._model.eval()
            
        epoch_loss = 0
        f1 = 0
            
        for i,(x_,y_) in enumerate(self._val_test_dl):
            x = x_.clone().detach().float().to(self.device)
            y = y_.clone().detach().float().to(self.device)
                
            (e,y_hat) = self.val_test_step(x,y)
            epoch_loss += e
            
            f1 += f1_score(y_hat.numpy(),y.numpy(),average='micro')
                
        print('Accuracy on validation set is: %',f1/i)
        
        return epoch_loss.item() / i
    
    def restore_last_session(self):
        if os.path.exists('checkpoints/'):
            try:
                print('Restoring Last Session!!!')
                c = len(os.listdir('checkpoints'))-1
                self.restore_checkpoint(c)
            except:
                print('Problem with restoring...')
        else:
            os.mkdir('checkpoints/')
            
    def get_lr(self):
        for param_group in self._optim.param_groups:
            return param_group['lr']
    
    def fit(self, epochs=-1):
        
        self.restore_last_session()
            
        trnLoss = [None]*epochs
        valLoss = [None]*epochs
        
        e = 0
         
        loss_train = np.empty((epochs,1))
        loss_val = np.empty((epochs,1))
        
        while e<epochs and self._early_stopping_cb.early_stop==False:
            print('Epoch :',e+1,'/',epochs,'\n')
            loss_train[e]= self.train_epoch()
            print('c loss: ',loss_train[e])
            loss_val[e] = self.val_test()
            print('v loss: ',loss_val[e])
            self._early_stopping_cb(loss_val[e])                
            self._shed.step()
            print(self.get_lr())
            self.save_checkpoint(e)
            e += 1

        return [loss_train,loss_val]            
        
        
        
