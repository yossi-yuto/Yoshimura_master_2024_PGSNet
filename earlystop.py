import os
from datetime import datetime

import torch
import numpy as np

''' ealy stop'''
class EarlyStopping:
  def __init__(self, patience=15, verbose=False, filepath: str = None):
    '''
    Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
    '''
    self.patiecce = patience 
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.path = filepath
    self.flag = False

  def __call__(self, val_loss, model):
    score = -val_loss
    self.flag = False
    if self.best_score is None:
      self.best_score = score
      self.checkpoint(val_loss, model)
    
    elif score < self.best_score:
      self.counter += 1
      if self.verbose:
        print(f"EarlyStopping counter:{self.counter} out of {self.patiecce}")
      
      if self.counter >= self.patiecce:
        self.early_stop = True
    
    else:
      self.best_score = score
      self.checkpoint(val_loss, model)
      self.flag = True
      self.counter = 0

  def checkpoint(self, val_loss, model):
    '''Saves model when validation loss decrease'''
    if self.verbose:
      print(f"Validation loss decreased({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
    torch.save(model.state_dict(), self.path)
    self.val_loss_min = val_loss



class EarlyStopping_Metrics:
    def __init__(self, patience=15, verbose=False, filepath: str = None):
        '''
        Args:
            patience (int): How long to wait after last time score improved.
                            Default: 15
            verbose (bool): If True, prints a message for each score improvement. 
                            Default: False
            filepath (str): Path for the checkpoint to be saved to.
                            Default: None
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = filepath

    def __call__(self, score, model):
        self.flag = False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score <= self.best_score: # スコアが改善されない場合
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.flag = True
            self.counter = 0

    def save_checkpoint(self, score, model):
        ''' Saves model when score improves. '''
        if self.verbose:
            print(f"Score improved ({self.best_score:.6f} --> {score:.6f}). Saving model ...")
        if self.path:
            torch.save(model.state_dict(), self.path)
