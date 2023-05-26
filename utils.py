# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:09:04 2023

@author: Revlis_user
"""
#%%
'Import Libraries'

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
from sklearn.metrics import precision_recall_curve
import torch
from torch.utils.data import Dataset

from params import *
from preprocessing import *

#%%
'Data Loader'

class Dataset(Dataset):
    
    # Random Crop Signal to Segment with w_size seconds length
    def __init__(self, data, labels, window_size,):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the full record and its corresponding label
        record = self.data[index]
        label = self.labels[index]

        # Randomly select a segment of fixed length from the full record
        start = random.randint(0, record.shape[-1] - self.window_size)
        segment = record[:,start : start + self.window_size]

        return segment, label

class StepWiseTestset(Dataset):
    
    # Testset for Element-Wise Maximum
    def __init__(self, data, labels, window_size, stride):
        
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # Get the full record and its corresponding label
        record = self.data[index]
        label = self.labels[index]

        # Create segments with overlapping windows
        segments = []
        start = 0
        while start + self.window_size <= record.shape[-1]:
            segment = record[:, start : start + self.window_size]
            segments.append(segment)
            start += self.stride

        return segments, label

#%%
'Beautiful Chart for Confusion Matrix'

mpl.style.use('seaborn')
def plot_cm(cm, classes, 
           normalize=False,
           title="Confusion Matrix",
           cmap=plt.cm.Purples):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis =1)[:, np.newaxis]
        print("Normalized CM")
    else:
        print('CM without Normalization')
        
    print(cm)
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, )
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                horizontalalignment='center',
                color="white" if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#%%
'Finding Threshold'

def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], 
                                                              y_score[:, k])
        # Compute f1 score with nan_to_num to avoid nans messing
        _score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return opt_precision, opt_recall, opt_threshold
