# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:52:22 2023

@author: Revlis_user
"""

#%%
'Import Libraries'

from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from params import *
from preprocessing import *
from utils import *

from xresnet1d import *

#%%
'Data Loader'

train_set = Dataset(train_x, train_y.values, int(w_size * model_sr),)
val_set = Dataset(val_x, val_y.values, int(w_size * model_sr))

train_loader = DataLoader(dataset=train_set, 
                          batch_size = batch_size, 
                          shuffle=True,)
val_loader = DataLoader(dataset=val_set, 
                        batch_size = batch_size, 
                        shuffle=True)

test_set = StepWiseTestset(test_x, test_y.values, 
                           int(w_size * model_sr), int(.5 * model_sr))

test_loader = DataLoader(dataset=test_set, 
                         batch_size = 1, 
                         shuffle=False)

#%%
'Training the Model'

model = xresnet1d152(model_drop_r=model_dropout, 
                     original_f_number=False,
                     # Original xResnet's Resblock Filter Dim if True
                     # PTB-XL Benchmark Article's Filter Dim if False
                     fc_drop=fc_drop,)
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr =.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.1, patience=5)
step_per_epoch = math.ceil(len(train_set)/batch_size)

train_loss = []
val_loss = []
train_f1 = []
val_f1 = []

for epoch in range(n_epochs):
    t0 = datetime.now()
    loss_per_step = []
    f1_per_step = []
    model.train()
    with tqdm(train_loader, unit='batch',) as per_epoch:
        for x,y in per_epoch:
            opt.zero_grad()
            per_epoch.set_description(f"Epoch: {epoch+1}/{n_epochs}")
            x,y = x.to(device, torch.float32), y.to(device, torch.float32)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            y_hat = (out_activation(y_hat).cpu().detach().numpy()>=.5
                     ).astype(int)
            loss_per_step.append(loss.item())
            score = f1_score(y.cpu().numpy().astype('int32'), y_hat,
                             average='samples', zero_division = 0)
            f1_per_step.append(score)
            loss.backward()
            opt.step()
            per_epoch.set_postfix(train_loss=loss.item(), 
                                  f1 = score)
        if scheduler.__module__ == 'torch.optim.lr_scheduler':
            scheduler.step(1)
    train_loss.append(sum(loss_per_step)/len(loss_per_step))
    train_f1.append(np.average(f1_per_step))
    
    val_loss_per_step = []
    val_f1_per_step = []
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, unit='batch',) as per_val_epoch:
            for x_val,y_val in per_val_epoch:
                per_val_epoch.set_description("Model Evaluation: ")
                x_val,y_val = x_val.to(device, torch.float32
                                       ), y_val.to(device, 
                                                   torch.float32)
                y_hat_val = model(x_val)
                loss_val = loss_fn(y_hat_val, y_val)
                y_hat_val = (out_activation(
                    y_hat_val).cpu().detach().numpy()>=.5).astype(int)
                val_loss_per_step.append(loss_val.item())
                score = f1_score(y_val.cpu().numpy().astype('int32'), 
                                 y_hat_val,average='samples', 
                                 zero_division = 0)
                val_f1_per_step.append(score)
                per_val_epoch.set_postfix(val_loss=loss_val.item(), 
                                          f1 = score)
    val_loss.append(sum(val_loss_per_step)/len(val_loss_per_step))
    val_f1.append(np.average(val_f1_per_step))
    
    torch.save(model.state_dict(), 
               f'results/model_weight_{epoch+1:02d}.pt')
    
    dt = datetime.now() - t0
    with open('results/log.txt',
              "a+") as external_file:
        print(f'''train_loss: {train_loss[-1]:.6f}, 
              train_f1: {train_f1[-1]:.6f}''',
              f"val_loss: {val_loss[-1]:.6f}, val_f1: {val_f1[-1]:.6f}",
              f'Time Duration: {dt}',
              file=external_file)
    print(f'train_loss: {train_loss[-1]:.6f}, train_f1: {train_f1[-1]:.6f}',
          f"val_loss: {val_loss[-1]:.6f}, val_f1: {val_f1[-1]:.6f}",
          f'Time Duration: {dt}')

#%%
'Results Visualization'

plt.figure()
plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.legend()
plt.show()
plt.savefig('results/Loss.png', )

plt.figure()
plt.plot(train_f1, label='train F1')
plt.plot(val_f1, label='val F1')
plt.legend()
plt.show()
plt.savefig('results/F1_Score.png', )

predictions = []

model.eval()
with torch.no_grad():
    for x_test, y_test in test_loader:
        windows_predictions = []
        for segments in x_test:
            segments = segments.to(device, torch.float32)
            output = out_activation(model(segments))
            windows_predictions.append(output)
        windows_predictions = torch.stack(windows_predictions)
        aggregated_prediction = torch.max(windows_predictions, dim=0)[0]
        predictions.append(aggregated_prediction)
        
predictions = torch.cat(predictions, dim=0)        
predictions = predictions.cpu().detach().numpy()

'Compute confusion matrix with opt_threshold'

threshold = get_optimal_precision_recall(test_y.values,
                                         predictions)[-1]
threshold = dict(zip(output_label, threshold))

with open('results/log.txt',
          "a+") as external_file:
    print(f'threshold: {threshold}', 
          file=external_file)
    external_file.close()

for i in range(len(output_label)):
    cm = confusion_matrix(test_y.iloc[:,i].values, 
                          predictions[:,i]>=threshold[output_label[i]])
    plt.figure()
    plot_cm(cm, classes = list(range(cm.shape[0])),
            title = f'CM_{output_label[i]}')
    plt.savefig(f'results/CM_{output_label[i]}.png')
    
    score = f1_score(test_y.iloc[:,i].values, 
                     predictions[:,i]>=threshold[output_label[i]], 
                     zero_division = 0)
    with open('results/log.txt',
          "a+") as external_file: 
        print(f'{output_label[i]}_F1_Score: {score}', 
          file=external_file)
    external_file.close()
