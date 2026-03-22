# ----------------------------------------------------------------------
# Code copied from GitHub repo: hbprosper/mlinphysics
# Harrison B. Prosper
# Created: Sun Mar 22 2026
# ----------------------------------------------------------------------
import os, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td

import time
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
try:
    import scipy.stats as st
except:
    raise ImportError('''
    Please install scipy:

        conda install scipy
    ''')

import yaml
# ----------------------------------------------------------------------
# Simple utilities
# ----------------------------------------------------------------------
def number_of_parameters(model):
    '''
    Get number of trainable parameters in a model.
    '''
    return sum(param.numel() 
               for param in model.parameters() 
               if param.requires_grad)

def initialize_model(model, paramsfile):
    # load parameters of neural network and set to eval mode
    model.load_state_dict(torch.load(paramsfile, 
                                     weights_only=True,
                                     map_location=torch.device('cpu')))
    model.eval()

def initialize_parameters(model):
    for param in model.parameters(): 
        if param.requires_grad:
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
                nn.init.zeros_(param.bias)

# This function assumes that the len(loader) is the same as
# the batch size given when the loader is instantiated
def compute_avg_loss(objective, loader):
    objective.eval()
    avg_loss = sum([float(objective(x, y).detach().cpu()) for x, y in loader]) / len(loader)
    return avg_loss

def elapsed_time(now, start):
    etime = now() - start    
    t = etime
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    seconds = t - 60 * minutes
    etime_str = "%2.2d:%2.2d:%2.2d" % (hours, minutes, seconds)
    return etime_str, etime, (hours, minutes, seconds)
# -----------------------------------------------------------------------
# Classes 
# -----------------------------------------------------------------------
class ExponentialLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, f, y):
        """
        outputs:  shape (batch_size, 1)
        targets:  shape (batch_size, 1)
        """
        losses = torch.exp( -0.5 * y * f)
        return torch.mean(losses)

class Model(nn.Module):
    
    def __init__(self): 
        super().__init__()
        self.net = None
        
    def save(self, paramsfile):
        # save parameters of neural network
        torch.save(self.state_dict(), paramsfile)
    
    def load(self, paramsfile):
        # load parameters of neural network and set to eval mode
        self.load_state_dict(torch.load(paramsfile, 
                                        weights_only=True,
                                        map_location=torch.device('cpu')))
        self.eval()
            
    def forward(self, x, p=None):
        if p is not None:
            p = p.repeat(len(x), 1) if p.ndim < 2 else p
            x = torch.concat((x, p), dim=-1)
            
        if self.net == None:
            raise ValueError('self.net not defined. Please do so in constructor!')
            
        y = self.net(x)   
        return y
        
class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class ResNet(nn.Module):
    '''
    ResNet(n_input : int, n_width : int)
    '''
    def __init__(self, n_input, n_width, f_hidden=Sin()):
        # remember to initialize base (that is, parent) class
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_input, n_width), f_hidden,
                                 nn.Linear(n_width, n_input), f_hidden)
    def forward(self, x):
        return self.net(x) + x

class FCNN(Model):
    '''
    Model a fully-connected neural network (FCNN).
    '''
    
    def __init__(self, 
                 n_inputs, 
                 n_hidden=4, 
                 n_width=32, 
                 f_hidden=Sin, 
                 f_output=None):
        
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_width  = n_width
        self.f_hidden = f_hidden
        self.f_output = f_output
        
        cmd  = 'nn.Sequential(nn.Linear(n_inputs, n_width), f_hidden(), '
        cmd += ', '.join(['nn.Linear(n_width, n_width), f_hidden()' 
                          for _ in range(n_hidden-1)])
        if f_output:
            cmd += ', nn.Linear(n_width, 1), f_output())'
        else:
            cmd += ', nn.Linear(n_width, 1))'
        cmd  = cmd.replace(', ,', ', ') # Hack!
        
        self.net = eval(cmd)

class Config:
    '''
        Manage simple ML application configuration

          name:      name stub for all files, including the yaml file
          batchsize: 
          base_lr:   base learning rate
            :
          etc.
    '''
    def __init__(self, name, mkdir=True, dirname=None, verbose=0):
        '''
        name  : string   Stub for all files, including the yaml file, or 
                         the name of a yaml file. A json file is identified 
                         by the extension .yaml
                
                            1. if name is a name stub, create a new yaml object.
                            2. if name is a yaml filename, create the yaml object
                               from the file.
                               
        mkdir : bool     If True create log folder [True]. The default name is
                         runs/<timestamp>.
                         
        dirname : string If given use this as the name of the folder: 
                         runs/<dirname>.
        '''
        self.makedir = mkdir
        self.dirname = dirname
        if self.dirname is None:
            self.time = time.ctime()
            self.dirname = datetime.now().strftime("%Y-%m-%d_%H%M")
            
        logdir = f"runs/{self.dirname}"
        self.logdir = logdir
        
        # create run folder if self.makedir is True
        self.mkdir()
                
        # check if a yaml file has been specified
        if name.endswith('.yaml') or name.endswith('.yml'):
            self.cfg_filename = name # cache filename
            self.load(name)
        else:
            # this not a yaml file specification, assume it is a name stub
            # and build a Python dictionary that specifies the structure of
            # 
            self.cfg = {}
            cfg = self.cfg
            
            cfg['name'] = name
    
            # construct output file names    
            o_cfg = {}

            o_cfg['losses']     = f'{logdir}/{name}_losses.csv'
            o_cfg['params']     = f'{logdir}/{name}_params.pth'
            o_cfg['init_params']= f'{logdir}/{name}_init_params.pth'
            o_cfg['plots']      = f'{logdir}/{name}_plots.png'

            cfg['file'] = o_cfg
    
            # create a default name for yaml configuration file
            # this name will be used if a filename is not
            # specified in the save method
            self.cfg_filename = f'{logdir}/{name}_config.yaml'
    
        if verbose:
            print(self.__str__())

    def mkdir(self):
        if self.makedir:
            os.makedirs("runs", exist_ok=True)
            os.makedirs(self.logdir, exist_ok=True) 
        
    def load(self, filename):
        # make sure file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f'{filename}')
        
        # read yaml file and cache as Python dictionary
        with open(filename, mode="r") as file:
            self.cfg = yaml.safe_load(file)

    def save(self, filename=None):
        # if no filename specified use default filename
        if filename == None:
            filename = self.cfg_filename

        # require .yaml extension
        if not (filename.endswith('.yaml') or filename.endswith('.yml')):
            raise NameError('the output file must have extension .yaml')
            
        # save to yaml file
        open(filename, 'w').write(self.__str__())
        
    def __call__(self, key, value=None):
        '''
        Return the value of the specified key.

        Notes
        -----
        1. If the key is in the dictionary and value is specified then 
        update the value of the key and return the value, otherwise 
        return the existing value of the key.

        2. If the key is not in the dictionary add it to the dictionary with
        the specified value and return the value. If no value is given raise 
        a KeyError exception.
        '''
        # this method can be used to fill out the rest
        # of the Python dictionary
        keys = key.split('/')
        
        # if key exists and value !=None update the value
        # else return its value
        cfg = self.cfg
        
        for ii, lkey in enumerate(keys):
            depth = ii + 1
            
            if lkey in cfg:
                # key is in dictionary
                
                val = cfg[lkey]
                if depth < len(keys):
                    # recursion
                    cfg = val
                else:
                    if type(value) == type(None):
                        # key exists and no value has been specified
                        # so return existing value
                        value = val
                    else:
                        # key exists and a value has been specified
                        # so update key and return new value
                        cfg[key] = value # update value
                    break
            else:
                # key is not in dictionary object, so add it
                
                if value == None:
                    # no value specified, so we can't add this key
                    raise KeyError(f'key "{lkey}" not found')
                    
                elif depth < len(keys):
                    cfg[lkey] = {}
                    cfg = cfg[lkey]
                else:
                    try:
                        cfg[lkey] = value
                    except:
                        pkey = keys[ii-1]
                        print(
                            f'''
    Warning: key '{key}' not created because '{pkey}' is 
    of type {str(type(pkey))}
                        ''')
        return value

    def __str__(self):
        # return a pretty printed string of the yaml object (help from ChatGPT)
        return str(yaml.dump(
            self.cfg,                 
            sort_keys=False,           # keep key order
            default_flow_style=False,  # use block style 
            indent=1,                  # indentation level
            allow_unicode=True))
# ------------------------------------------------------------------------
class LRStepScheduler:
    def __init__(self, 
                 optimizer, n_steps, n_iters_per_step, base_lr, gamma, 
                 verbose=True):
        
        self.optimizer = optimizer
        self.scheduler = self.__get_steplr_scheduler(
            optimizer, n_steps, n_iters_per_step, base_lr, gamma
        )
        self.verbose   = verbose
        self.curr_lr   = -1.0

    def __get_steplr_scheduler(self,
        optimizer, n_steps, n_iters_per_step, base_lr, gamma):
        
        # Number of milestones in multistep LR schedule
        n_milestones = n_steps - 1
        print(f'number of milestones: {n_milestones:10d}\n')
    
        # Learning rate milestones
        milestones = [n * n_iters_per_step for n in range(n_steps)]
    
        # learning rates
        lrs = [base_lr * gamma**i for i in range(n_steps)]
        
        print("Step | Milestone | LR")
        print("-----------------------------")
        for i in range(n_steps):
            print(f"{i:>4} | {milestones[i]:>9} | {lrs[i]:<10.1e}")
            if i < 1:
                print("-----------------------------")
        print()
        
        # drop first entry of milestones list because it contains the base LR    
        return MultiStepLR(optimizer, milestones=milestones[1:], gamma=gamma)

    def step(self):
        self.scheduler.step()
        
    def lr(self):
        lrate = self.optimizer.param_groups[0]['lr']
        if lrate != self.curr_lr:
            self.curr_lr = lrate
            if self.verbose:
                print()
                print(f'\t\tlearning rate: {lrate:10.3e}')
        return lrate
# -----------------------------------------------------------------------
class Objective(nn.Module):

    def __init__(self, model, avgloss):
        super().__init__()
        self.model = model
        self.avgloss = avgloss
        
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save(self, paramsfile):
        self.model.save(paramsfile)
    
    def forward(self, x, y):
        f = self.model(x)
        return self.avgloss(f, y)
 
