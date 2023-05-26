# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:27:12 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import os
import torch
import torch.nn as nn

#%%
'Hyperparameters'

activation = nn.LeakyReLU(inplace=True)
w_size = 2.5 # Random Segmentation Size (Unit: seconds)
model_sr = 100 #Sampling Rate for Model Input
stem_k, block_k = 7, 5 # Kernel Size for Stem and ResBlock Conv1d
data_dim = 12 # Dimension for Model Input
batch_size = 256
model_dropout = None # xResNet Dropout Rate
fc_drop = None # FC Layer Dropout Rate
out_activation = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss()
n_epochs = 120

#%%
"Setting GPU"

use_cpu = False
m_seed = None # Set Seed for Reproducibility

if use_cpu:
    device = torch.device('cpu')

elif torch.cuda.is_available(): 
    device = torch.device('cuda')
    if m_seed:
        torch.cuda.manual_seed(m_seed)
    torch.cuda.empty_cache()

elif not torch.backends.mps.is_available(): #Setting GPU for Mac User
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was "
              "NOT built with MPS enabled.")
    else:
        print("MPS not available because this MacOS version is NOT 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
print(device)

#%%
'Make Output Dir'

if not os.path.isdir('results'):
    os.makedirs('results')
