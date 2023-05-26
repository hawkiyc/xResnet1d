# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:27:23 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import * # Import Scaler You Need

from params import *

#%%
'Data Preprocessing'

def preprocess_signals(train, val, test):

    'Preprocessing Method You Are Going to Use'
    return processed_train, processed_val, processed_test

#%%
"Load Your Time Series Data"

train_x, train_y = np.load('file_name'), pd.read_csv('file_name')
test_x, test_y = np.load('file_name'), pd.read_csv('file_name')
out_dim = test_y.values.shape[-1] # Dim for Model Output
output_label = list(test_y.columns)

#%%
'Split Data'

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, 
                                                  test_size=.1, 
                                                  random_state=42)

train_x, val_x, test_x = preprocess_signals(train_x, val_x, test_x,)
