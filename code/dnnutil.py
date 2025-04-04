import os, sys
import numpy as np
import pandas as pd

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy.optimize as op
import torch
import torch.nn as nn
import time

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler
from glob import glob
from matplotlib.animation import FuncAnimation
#------------------------------------------------------------------------------
DELAY = 10 # seconds - interval between plot updates
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ncomputational device: %s\n" % DEVICE)
#------------------------------------------------------------------------------
class TimeLeft:
    '''
    Return the amount of time left.
    
    timeleft = TimeLeft(N)
    
    N: maximum loop count
    
      for i in timeleft:
          : :

    or
       timeleft(i, extra)
      
    '''
    def __init__(self, N):
        self.N = N        
        self.timenow = time.time
        self.start = self.timenow()
        self.str = ''
        
    def __del__(self):
        pass
    
    def __timestr(self, ii):
        
        # elapsed time since start
        elapsed = self.timenow() - self.start
        s = elapsed
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s = s - 60*m
        hh= h
        mm= m
        ss= s
        
        # time/loop
        count = ii+1
        t = elapsed / count
        f = 1/t
        
        # time left
        s = t * (self.N - count)
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s =  s - 60*m
        percent = 100 * count / self.N

        return "%10d|%6.2f%s|%2.2d:%2.2d:%2.2d/%2.2d:%2.2d:%2.2d|%6.1f it/s" % \
            (count, percent, '%', hh, mm, ss, h, m, s, f)
        
    def __iter__(self):
        
        for ii in range(self.N):
            
            if ii < self.N-1:
                print(f'\r{self.__timestr(ii):s}', end='')
            else: 
                print(f'\r{self.__timestr(ii):s}')
                
            yield ii
            
    def __call__(self, ii, extra='', colorize=False):
        
        if extra != '':
            if colorize:
               extra = "\x1b[1;34;48m|%s\x1b[0m" % extra
                
        self.a_str = f'{self.__timestr(ii):s}{extra:s}'
        
        if ii < self.N-1:
            print(f'\r{self.a_str}', end='')
        else:
            print(f'\r{self.a_str}')

    def __str__(self):
        return self.a_str
#------------------------------------------------------------------------------
# The loss file should be a simple text file with two columns of numbers:
#
#   train-losses,validation-losses
#------------------------------------------------------------------------------
def get_losses(loss_file):
    try:
        losses = pd.read_csv(loss_file).to_numpy()
        return losses[:, 0], losses[:, 1], losses[:, 2]
    except:
        return None

def get_timeleft(timeleft_file):
    try:
        return open(timeleft_file, 'r').read().strip()
    except:
        return None
#------------------------------------------------------------------------------
class Monitor:

    def __init__(self, loss_file='losses.csv'):

        self.loss_file = loss_file
        self.timeleft_file = loss_file.replace('losses.csv', 
                                               'timeleft.txt')
        if self.loss_file == self.timeleft_file:
            self.timeleft_file = 'timeleft.txt'
            
        print('loss file:     ', self.loss_file)
        print('timeleft file: ', self.timeleft_file)
        print()

        # set up an empty figure
        self.fig = plt.figure(figsize=(6, 4))
        self.fig.suptitle(self.loss_file)
        
        # add a subplot to it
        nrows, ncols, index = 1,1,1
        self.ax = self.fig.add_subplot(nrows, ncols, index)

    def __call__(self):
        interval = 1000 * DELAY # milliseconds
        self.ani = FuncAnimation(fig=self.fig, 
                            func=self.plot_losses, 
                            interval=interval, 
                            repeat=False, 
                            cache_frame_data=False)
        plt.show()

    def plot_losses(self, frame):
        ax, fig = self.ax, self.fig
        loss_file, timeleft_file = self.loss_file, self.timeleft_file
        
        ax.clear()
        ax.set_xlabel('iteration', fontsize=16)
        ax.set_ylabel('<loss>', fontsize=16)
        ax.grid(True, which="both", linestyle='-')
        fig.tight_layout()
        
        data = get_losses(loss_file)
        
        if type(data) != type(None):
            iters, train_losses, valid_losses = data    
            if train_losses[0]/train_losses[-1] > 25:
                ax.set_yscale('log')
    
            timeleft = get_timeleft(timeleft_file)
            if timeleft != None:
                ax.set_title(timeleft, fontsize=9)
            else:
                ax.set_title('iteration: %5d | %s' % (iters[-1], time.ctime()))
                
            ax.plot(iters, train_losses, c='red',  label='training')
            ax.plot(iters, valid_losses, c='blue', label='validation')
            ax.legend()
#------------------------------------------------------------------------------   
def split_data(data,
               test_fraction, 
               validation_fraction):

    # Split data into a part for training and a part for testing
    train_data, test_data = train_test_split(data, 
                                         test_size=test_fraction, 
                                         shuffle=True)

    # Split the training data into a part for training (fitting) and
    # a part for validating the training.
    v_fraction = validation_fraction * len(data) / len(train_data)
    train_data, valid_data = train_test_split(train_data, 
                                          test_size=v_fraction,
                                          shuffle=True)

    # reset the indices in the dataframes and drop the old ones
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)
    
    return train_data, valid_data, test_data 

def split_source_target(df, source, target):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    x = np.array(df[source])
    t = np.array(df[target])
    return x, t

# return a batch of data for the next step in minimization
def get_batch(x, t, batch_size, ii=-1):
    if ii <= 0:
        # selects at random "batch_size" integers from 
        # the range [0, batch_size-1] corresponding to the
        # row indices of the training data to be used
        rows = torch.randint(0, len(x)-1, size=(batch_size,))
        return x[rows], t[rows]
    else:
        jj = ii % batch_size
        start = jj * batch_size
        end = start + batch_size
        return x[start: end], t[start: end]

# Note: there are several average loss functions available 
# in pytorch, but it's useful to know how to create your own.
def average_quadratic_loss(f, t, x=None):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)

def average_quadratic_loss_weighted(f, t, x=None):
    # f and t must be of the same shape
    select = t > 0 # exclude instances with t <= 0
    f = f[select]
    t = t[select]
    return  torch.mean((f - t)**2 / t)

def average_cross_entropy_loss(f, t, x=None):
    # f and t must be of the same shape
    loss = torch.where(t > 0.5, torch.log(f), torch.log(1 - f))
    return -torch.mean(loss)

def average_quantile_loss(f, t, x):
    # f and t must be of the same shape
    tau = x.T[-1] # last column is tau.
    return torch.mean(torch.where(t >= f, 
                                  tau * (t - f), 
                                  (1 - tau)*(f - t)))

def average_median_loss(f, t, x=None):
    # f and t must be of the same shape
    return torch.mean(torch.where(t >= f, (t - f)/2, (f - t)/2))

def average_median_loss_weighted(f, t, x=None):
    # f and t must be of the same shape
    select = t > 0 # exclude instances with t <= 0
    f = f[select]
    t = t[select]
    return torch.mean(w*torch.where(t >= f, w*(t - f)/2, w*(f - t)/2))
    
# function to validate model during training.
def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and 
        # remember to reshape!
        outputs = model(inputs).reshape(targets.shape)
    return avloss(outputs, targets, inputs)
        
def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
def train(model, optimizer, dictfile, 
          avloss, getbatch,
          train_data, valid_data, 
          features, target,
          batch_size,
          n_iterations,
          device=DEVICE,
          lossfile='losses.csv',
          change=0.05,
          step=100):

    timeleftfile = lossfile.replace('losses.csv', 'timeleft.txt')   
    n = len(valid_data)

    train_x, train_t = split_source_target(train_data, features, target)
    valid_x, valid_t = split_source_target(valid_data, features, target)

    # load data onto computational device
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        train_x = torch.from_numpy(train_x).float().to(device)
        train_t = torch.from_numpy(train_t).float().to(device)
    
        valid_x = torch.from_numpy(valid_x).float().to(device)
        valid_t = torch.from_numpy(valid_t).float().to(device)

    # place model on current computational device
    model.to(device)
    
    n = len(valid_x)
    
    # start saving best model after the following number of iterations.
    start_saving = n_iterations // 100
    min_avloss   = float('inf')

    # initialize loss file
    if not os.path.exists(lossfile):
        open(lossfile, 'w').write('iteration,t_loss,v_loss\n')  
    df = pd.read_csv(lossfile)
    if len(df) < 1:
        xx = 0
    else:
        xx = df.iteration.iloc[-1]

    # enter training loop
    timeleft = TimeLeft(n_iterations)
    
    for ii in range(n_iterations):
            
        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        model.train()
        
        # get a random sample (a batch) of data
        x, t = getbatch(train_x, train_t, batch_size)

        def closure():
            optimizer.zero_grad()       # clear previous gradients
            # compute the output of the model for the batch of data x
            # -------------------------------------------------------
            # for the tensor operations with outputs and t to work
            # correctly, it is necessary that they be of the same
            # shape. We can do this with the reshape method.
            outputs = model(x).reshape(t.shape)
            
            # compute a noisy approximation to the average loss
            empirical_risk = avloss(outputs, t, x)
            
            # use automatic differentiation to compute a 
            # noisy approximation of the local gradient
    
            empirical_risk.backward()   # compute gradients
            return empirical_risk
            
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step(closure)            # move one step
        
        if ii % step == 0:
            
            t_loss = validate(model, avloss, train_x[:n], train_t[:n]).item() 
            v_loss = validate(model, avloss, valid_x[:n], valid_t[:n]).item()

            # write to loss file
            open(lossfile, 'a').write(f'{xx:12d},{t_loss:12.6},{v_loss:12.6}\n')

            # save model to file if there has been a significant change
            if v_loss < (1 - change) * min_avloss:
                min_avloss = v_loss
                if ii > start_saving:
                    torch.save(model.state_dict(), dictfile)

            line = f'|{xx:12d}|{t_loss:12.6f}|{v_loss:12.6f}|'
            if ii < step:
                print(line)
            else:
                timeleft(ii, line)
                open(timeleftfile, 'w').write(f'{str(timeleft):s}\n')
            
            xx += step
       
    print()
    return

def plot_average_loss(lossfile, ftsize=18, xlabel='Iterations', filename='fig_losses.pdf'):
    
    losses = get_losses(lossfile)
    if type(losses) == type(None):
        print('** loss file not found **\n')
        return
        
    xx, yy_t, yy_v = losses
    
    # create an empty figure
    fig = plt.figure(figsize=(5, 4))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')

    ax.set_xlabel(xlabel, fontsize=ftsize)
    ax.set_ylabel('average loss', fontsize=ftsize)
    if xlabel.lower()[0] == 'i':
        ax.set_xscale('log')
        
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

class AveragedModel:
    
    def __init__(self, model, scale=0.02, size=25):
        from glob import glob
        from numpy import random
        from copy import deepcopy
        
        self.models = []
        self.models.append(model)
        state_dict = model.state_dict()

        for ii in range(1, size):
            
            # make a deep copy of input model            
            self.models.append(deepcopy(model))
            
            with torch.no_grad():
                
                for name, param in self.models[-1].named_parameters():
                    
                    if param.requires_grad:
                        x = state_dict[name]
                        y = random.normal(x, scale)
                        param.copy_(torch.Tensor(y))

    def __call__(self, x):
        self.models[0].eval()
        y = self.models[0](x)
        for m in self.models[1:]:
            y += m(x)
        y /= len(self.models)
        return y

    def eval(self):
        for m in self.models:
            m.eval()
