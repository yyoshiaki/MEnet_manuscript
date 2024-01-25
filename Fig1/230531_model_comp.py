#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import os
from pickle import dump
from pickle import load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(100)

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import sys
import yaml
import time


# In[2]:


# f = open(sys.argv[1], 'r+')
f = open('../params/230531_compare_algorithms_test_indsamples.yaml', 'r+')
seed = 0

dict_input = yaml.load(f)
os.makedirs(dict_input['output_dir'] + "/{}".format(seed), exist_ok=True)

f_selected = dict_input['ref_train_bin']
f_integrated = dict_input['integrated']
f_pickle_train = dict_input['pickle_train']
f_train_label = dict_input['f_train_label']
f_pickle_test = dict_input['pickle_test']
f_test_label = dict_input['f_test_label']
f_pickle_test_mix = dict_input['pickle_test_mix']
f_test_label_mix = dict_input['f_test_mix_label']
n_mix = dict_input['n_mix']
f_train = dict_input['ref_train']
f_all = dict_input['ref_all']
dir_output = dict_input['output_dir']
min_train_count = dict_input['min_train_count']
max_test_count = dict_input['max_test_count']

fill = 'mean'


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(f_train_label) & os.path.exists(f_test_label):
    df_ref_train = pd.read_csv(f_train_label, index_col=0)
    df_ref_test = pd.read_csv(f_test_label, index_col=0)
    print('reference labels loaded')

else:
    df_ref = pd.read_csv(f_selected)
    # df_ref.head()

    df_ref_train = pd.read_csv(f_train)
    df_ref_all = pd.read_csv(f_all)

    df_ref_all = df_ref_all[df_ref_all['Assay'] == 'WGBS']
    df_ref_train = df_ref_train[~df_ref_train.MinorGroup.isna()]
    df_ref_all = df_ref_all[~df_ref_all.MinorGroup.isna()]

    list_cat = list(df_ref_train['Tissue'].value_counts()[df_ref_train['Tissue'].value_counts() >= min_train_count].index)
    df_ref_all = df_ref_all[df_ref_all['Tissue'].isin(list_cat)]

    df_ref_train = df_ref_all[df_ref_all.Project.isin(df_ref_train['Project'].unique())]
    df_ref_test = df_ref_all[~df_ref_all.Project.isin(df_ref_train['Project'].unique())]

    def subsample_dataframe(df: pd.DataFrame, col: str, max_test_count: int):
        # get unique values and their counts
        value_counts = df[col].value_counts()

        # identify those values where count is more than max_test_count
        over_max = value_counts[value_counts > max_test_count].index

        subsampled_df = df.loc[~df[col].isin(over_max)]

        for val in over_max:
            subsampled_rows = df.loc[df[col] == val].sample(max_test_count)
            subsampled_df = pd.concat([subsampled_df, subsampled_rows])
        
        return subsampled_df

    df_ref_test = subsample_dataframe(df_ref_test, 'Tissue', max_test_count)

    assert len(set(df_ref_train.FileID) & set(df_ref_test.FileID)) == 0

    df_ref_train.to_csv(f_train_label)
    df_ref_test.to_csv(f_test_label)
    print('train\n', df_ref_train.Tissue.value_counts())
    print('test\n', df_ref_test.Tissue.value_counts())

if os.path.exists(f_pickle_train) & os.path.exists(f_pickle_test):
    df_train = pd.read_pickle(f_pickle_train)
    df_test = pd.read_pickle(f_pickle_test)
    print('pickle loaded')

else:
    print('no processed pickle file. Now generating inputs...')
    df_selected =         pd.read_csv(f_selected, index_col=0)
    df_selected = df_selected.drop_duplicates()
    # df_selected.head()

    df_all = pd.read_csv(f_integrated, index_col = 0)
    # df_all.head()

    df_all.index = df_all['chr'] + ':' + df_all['start'].astype(str) + '-' + df_all['end'].astype(str)
    df = df_all.reindex(df_selected.index)
    df_all = None

    df_train = df[['rate_' + x for x in df_ref_train.FileID]]
    df_test = df[['rate_' + x for x in df_ref_test.FileID]]
    df_train.columns = df_ref_train.FileID
    df_test.columns = df_ref_test.FileID

    df_train = df_train.dropna(axis=0, how='all')
    df_test = df_test.dropna(axis=0, how='all')

    list_index = list(set(df_test.index) & set(df_train.index))
    df_train = df_train.loc[list_index]
    df_test = df_test.loc[list_index]

    df_train.to_pickle(f_pickle_train)
    df_test.to_pickle(f_pickle_test)


# In[4]:


labels_train = pd.get_dummies(df_ref_train.Tissue)
labels_test = pd.get_dummies(df_ref_test['Tissue']).reindex(columns=labels_train.columns).fillna(0).astype(int)


if str(fill).isdigit():
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=fill)
elif fill in ['median', 'mean', 'most_frequent']:
    imp = SimpleImputer(missing_values=np.nan, strategy='median')

imp.fit(np.array(df_train).T)

def preprocess(X):
    X_imp = imp.transform(X)
    return X_imp

# x_train, x_test, y_train, y_test = train_test_split(
#      preprocess(np.array(df).T), np.array(labels), test_size=0.2, random_state=seed)
x_train = preprocess(np.array(df_train).T)
x_test = preprocess(np.array(df_test).T)
y_train = np.array(labels_train)
y_test = np.array(labels_test)

x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
# y_train = torch.tensor(y_train, dtype=torch.long).to(device)
x_test = torch.FloatTensor(x_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)
df_nnls_ref = pd.DataFrame(x_train.cpu().detach().numpy())


# In[5]:


x_train.shape


# In[6]:


y_train.shape


# In[7]:



class OneHotCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        return loss
    
criterion = OneHotCrossEntropy()

df_nnls_ref = pd.DataFrame(x_train.cpu().detach().numpy())
df_nnls_ref.columns = df_train.index
df_nnls_ref['Tissue'] =     [labels_train.columns[np.where(x==1)[0][0]] for x in y_train.cpu().detach().numpy()]
df_nnls_ref = df_nnls_ref.groupby(by='Tissue').mean().T
df_nnls_ref = df_nnls_ref.T.reindex(labels_train.columns).dropna().T
arr_x_test = preprocess(x_test.cpu().detach().numpy())

start = time.time()
list_nnls = []
for i in range(arr_x_test.shape[0]):
    list_nnls.append(nnls(preprocess(df_nnls_ref.T).T, arr_x_test[i,:])[0])
test_time = time.time() - start

df_pred_nnls = pd.DataFrame(list_nnls)
df_pred_nnls.columns = df_nnls_ref.columns
df_pred_nnls = df_pred_nnls.T.reindex(labels_train.columns).fillna(0).T
df_pred_nnls = df_pred_nnls.div(df_pred_nnls.sum(axis=1), axis=0)

after_train = criterion(torch.FloatTensor(np.array(df_pred_nnls)).to(device), y_test) 
print('loss of nnls' , after_train.item())


# In[8]:


class Mixup_dataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform=None, n_choise=5):
        self.transform = transform
        self.data_num = data.shape[0]
        self.data = data
        self.label = label
        self.channel = label.shape[1]
        self.n_choise = n_choise

#         if self.transform == 'mix':
# #             out_data = self.transform(out_data)
# #             print(data)
#             mix_rate = F.softmax(torch.rand(data.shape[0], data.shape[0])).to(device)
# #             print(mix_rate)
#             self.data = torch.matmul(data.T , mix_rate).T
#             self.label = torch.matmul(label.T , mix_rate).T

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform == 'mix':
            idx_rand = torch.multinomial(torch.ones(self.data.shape[0]), self.n_choise)
            out_data = self.data[idx_rand].mean(axis=0)
            out_label = self.label[idx_rand].mean(axis=0)

        else:
            out_data = self.data[idx]
            out_label =  self.label[idx]
        
        return out_data, out_label


# In[9]:


if os.path.exists(f_pickle_test_mix) & os.path.exists(f_test_label_mix):
    x_test_mix = pd.read_pickle(f_pickle_test_mix)
    y_test_mix = pd.read_csv(f_test_label_mix, index_col=0)
    
else:
    data_set_test = Mixup_dataset(x_test, y_test, transform='mix', n_choise=5)

    list_x_test_mix = []
    list_y_test_mix = []
    for _ in range(n_mix):
        _x_test, _y_test = next(iter(data_set_test))
        list_x_test_mix.append(list(_x_test.cpu().numpy()))
        list_y_test_mix.append(list(_y_test.cpu().numpy()))

    x_test_mix = pd.DataFrame(list_x_test_mix, columns=df_test.index).T
    y_test_mix = pd.DataFrame(list_y_test_mix, columns=labels_test.columns)

    x_test_mix.to_pickle(f_pickle_test_mix)
    y_test_mix.to_csv(f_test_label_mix)

x_test_mix = torch.FloatTensor(np.array(x_test_mix)).to(device).T
y_test_mix = torch.FloatTensor(np.array(y_test_mix)).to(device)


# In[10]:


print('NNLS')
df_nnls_ref = pd.DataFrame(x_train.cpu().detach().numpy())
# df_nnls_ref.columns = df_train.index
df_nnls_ref['Tissue'] =     [labels_train.columns[np.where(x==1)[0][0]] for x in y_train.cpu().detach().numpy()]
df_nnls_ref = df_nnls_ref.groupby(by='Tissue').mean().T
df_nnls_ref = df_nnls_ref.T.reindex(labels_train.columns).dropna().T
arr_x_test = preprocess(x_test.cpu().detach().numpy())

start = time.time()
list_nnls = []
for i in range(arr_x_test.shape[0]):
    list_nnls.append(nnls(preprocess(df_nnls_ref.T).T, arr_x_test[i,:])[0])
test_time = time.time() - start

df_pred_nnls = pd.DataFrame(list_nnls)
df_pred_nnls.columns = df_nnls_ref.columns
df_pred_nnls = df_pred_nnls.T.reindex(labels_train.columns).fillna(0).T
df_pred_nnls = df_pred_nnls.div(df_pred_nnls.sum(axis=1), axis=0)

after_train = criterion(torch.FloatTensor(np.array(df_pred_nnls)).to(device), y_test) 
print('Test loss of nnls' , after_train.item())

arr_x_test_mix = x_test_mix.cpu().numpy()

start = time.time()
list_nnls = []
for i in range(arr_x_test_mix.shape[0]):
    list_nnls.append(nnls(preprocess(df_nnls_ref.T).T, arr_x_test_mix[i,:])[0])
test_time = time.time() - start

df_pred_nnls = pd.DataFrame(list_nnls)
df_pred_nnls.columns = df_nnls_ref.columns
df_pred_nnls = df_pred_nnls.T.reindex(labels_train.columns).fillna(0).T
df_pred_nnls = df_pred_nnls.div(df_pred_nnls.sum(axis=1), axis=0)

after_train_mix = criterion(torch.FloatTensor(np.array(df_pred_nnls)).to(device), y_test_mix) 
print('Mix Test loss of nnls' , after_train_mix.item())


# ## ARIC

# In[13]:


from ARIC import *

# ARIC(mix_path="../other_tools/ARIC/data/demo/mix.csv", ref_path="../other_tools/ARIC/data/demo/ref.csv")


# In[78]:


pd.read_csv("../other_tools/ARIC/data/demo/ref.csv").head()


# In[80]:


df_nnls_ref.head()


# In[82]:


df_nnls_ref.to_csv('../results/230531_compare_algorithms_test_indsamples_min3/ref_nnls.csv')


# In[11]:


pd.DataFrame(arr_x_test).T.to_csv('../results/230531_compare_algorithms_test_indsamples_min3/x_test_nnls.csv')
pd.DataFrame(arr_x_test_mix).T.to_csv('../results/230531_compare_algorithms_test_indsamples_min3/x_test_mix_nnls.csv')


# In[89]:


ARIC(mix_path='../results/230531_compare_algorithms_test_indsamples_min3/x_test_nnls.csv', 
     ref_path='../results/230531_compare_algorithms_test_indsamples_min3/ref_nnls.csv',
     is_methylation=False,
     save_path='../results/230531_compare_algorithms_test_indsamples_min3/pred_ARIC_test.csv')


# In[98]:


pd.read_csv(f'{dir_output}/pred_ARIC_test.csv', index_col=0).T


# In[96]:


pd.DataFrame(y_test.cpu().numpy())


# In[100]:


after_train = criterion(torch.FloatTensor(np.array(pd.read_csv(f'{dir_output}/pred_ARIC_test.csv', index_col=0).T)).to(device), y_test)
print('Test loss of ARIC' , after_train.item())


# In[19]:


after_train_mix = mean_squared_error(pd.read_csv(f'{dir_output}/pred_ARIC_test.csv', index_col=0).T, y_test.cpu().detach().numpy())
print('Mix Test loss of ARIC' , after_train_mix.item())


# In[ ]:


pd.DataFrame(arr_x_test_mix).T.to_csv('../results/230531_compare_algorithms_test_indsamples_min3/x_test_mix_nnls.csv')


# In[14]:


ARIC(mix_path=f'{dir_output}/x_test_mix_nnls.csv', 
     ref_path=f'{dir_output}/ref_nnls.csv',
     is_methylation=False,
     save_path=f'{dir_output}/pred_ARIC_test_mix.csv')


# In[60]:


after_train_mix = mean_squared_error(pd.read_csv(f'{dir_output}/pred_ARIC_test_mix.csv', index_col=0).T, y_test_mix.cpu().detach().numpy())
print('Mix Test loss of ARIC' , after_train_mix.item())


# ## CIBERSORT

# In[11]:


df_nnls_ref.to_csv('../results/230531_compare_algorithms_test_indsamples_min3/ref_nnls.tsv', sep='\t')


# In[11]:


pd.DataFrame(arr_x_test).T.to_csv('../results/230531_compare_algorithms_test_indsamples_min3/x_test_nnls.tsv', sep='\t')
pd.DataFrame(arr_x_test_mix).T.to_csv('../results/230531_compare_algorithms_test_indsamples_min3/x_test_mix_nnls.tsv', sep='\t')


# In[17]:


pd.read_csv('../other_tools/Cibersort/CIBERSORT-Results.txt', sep='\t', index_col=0)[labels_train.columns]


# In[23]:


pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_CIBERSORT.csv',
                                                index_col=0)


# In[24]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_CIBERSORT.csv',
                                               index_col=0)[labels_train.columns], 
                                     y_test.cpu().detach().numpy())
print('Test loss of CIBERSORT' , after_train_mix.item())


# In[25]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_CIBERSORT.csv',
                                               index_col=0)[labels_train.columns], 
                                     y_test_mix.cpu().detach().numpy())
print('Test Mix loss of CIBERSORT' , after_train_mix.item())


# ## EpiDISH

# In[26]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.CP.csv', index_col=0), 
                                     y_test.cpu().detach().numpy())
print('Test loss of CP' , after_train_mix.item())


# In[27]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.CP.csv', index_col=0), y_test_mix.cpu().detach().numpy())
print('Mix Test loss of CP' , after_train_mix.item())


# In[42]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.CBS.csv', index_col=0), 
                                     y_test.cpu().detach().numpy())
print('Test loss of CBS' , after_train_mix.item())


# In[43]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.CBS.csv', index_col=0), y_test_mix.cpu().detach().numpy())
print('Mix Test loss of CBS' , after_train_mix.item())


# In[29]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.RPC.csv', index_col=0), 
                                     y_test.cpu().detach().numpy())
print('Test loss of RPC' , after_train_mix.item())


# In[16]:


after_train_mix = mean_squared_error(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv', index_col=0), y_test_mix.cpu().detach().numpy())
print('Mix Test loss of RPC' , after_train_mix.item())


# In[ ]:





# In[ ]:





# In[ ]:





# ## Visualization

# In[14]:


y_pred = pd.read_csv(f'{dir_output}/pred_ARIC_test.csv', index_col=0).T


# In[62]:


color = 'inferno'

ind = pd.DataFrame(y_test.cpu().numpy()).sort_values(by=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], ascending=False).index

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

fig.suptitle('Purified')

sns.heatmap(pd.DataFrame(y_test.cpu().numpy(), columns=y_pred.columns).loc[ind], 
            yticklabels=[],  ax=axes[0,0], vmin=0, vmax=1, cmap=color)
axes[0,0].set_title('Ground Truth')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_w_Mixup_test.csv', index_col=0).loc[ind],
            yticklabels=[], ax=axes[0,1], vmin=0, vmax=1, cmap=color)
axes[0,1].set_title('NN with Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_wo_Mixup_test.csv', index_col=0).loc[ind], 
            yticklabels=[], ax=axes[0,2], vmin=0, vmax=1, cmap=color)
axes[0,2].set_title('NN without Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_ARIC_test.csv', index_col=0).T.loc[[str(x) for x in ind]], 
            yticklabels=[], ax=axes[0,3], vmin=0, vmax=1, cmap=color)
axes[0,3].set_title('ARIC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_CIBERSORT.csv', index_col=0).loc[ind, y_pred.columns], 
            yticklabels=[], ax=axes[1,0], vmin=0, vmax=1, cmap=color)
axes[1,0].set_title('CBS')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.RPC.csv', index_col=0).loc[['X'+str(x) for x in ind]], 
            yticklabels=[], ax=axes[1,1], vmin=0, vmax=1, cmap=color)
axes[1,1].set_title('RPC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.CP.csv', index_col=0).loc[['X'+str(x) for x in ind]], 
            yticklabels=[], ax=axes[1,2], vmin=0, vmax=1, cmap=color)
axes[1,2].set_title('CP')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NNLS_test.csv', index_col=0).loc[ind], 
            yticklabels=[],  ax=axes[1,3], vmin=0, vmax=1, cmap=color)
axes[1,3].set_title('NNLS')

plt.tight_layout()

plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/heatmap_test.pdf', bbox_inches='tight')


# In[64]:


ind = pd.DataFrame(y_test_mix.cpu().numpy()).sort_values(by=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], ascending=False).index

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

fig.suptitle('Mixed')

sns.heatmap(pd.DataFrame(y_test_mix.cpu().numpy(), columns=y_pred.columns).loc[ind], 
            yticklabels=[],  ax=axes[0,0], vmin=0, cmap=color)
axes[0,0].set_title('Ground Truth')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_w_Mixup_test_mix.csv', index_col=0).loc[ind],
            yticklabels=[], ax=axes[0,1], vmin=0, cmap=color)
axes[0,1].set_title('NN with Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_wo_Mixup_test_mix.csv', index_col=0).loc[ind], 
            yticklabels=[], ax=axes[0,2], vmin=0, cmap=color)
axes[0,2].set_title('NN without Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_ARIC_test_mix.csv', index_col=0).T.loc[[str(x) for x in ind]], 
            yticklabels=[], ax=axes[0,3], vmin=0, cmap=color)
axes[0,3].set_title('ARIC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_CIBERSORT.csv', index_col=0).loc[ind, y_pred.columns], 
            yticklabels=[], ax=axes[1,0], vmin=0, cmap=color)
axes[1,0].set_title('CBS')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv', index_col=0).loc[['X'+str(x) for x in ind]], 
            yticklabels=[], ax=axes[1,1], vmin=0, cmap=color)
axes[1,1].set_title('RPC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.CP.csv', index_col=0).loc[['X'+str(x) for x in ind]], 
            yticklabels=[], ax=axes[1,2], vmin=0, cmap=color)
axes[1,2].set_title('CP')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NNLS_test_mix.csv', index_col=0).loc[ind], 
            yticklabels=[], ax=axes[1,3], vmin=0, cmap=color)
axes[1,3].set_title('NNLS')

plt.tight_layout()

plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/heatmap_test_mix.pdf', bbox_inches='tight')


# In[65]:


ind = pd.DataFrame(y_test_mix.cpu().numpy()).sort_values(by=[2,3,4,10], ascending=False).index

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

fig.suptitle('Mixed')

sns.heatmap(pd.DataFrame(y_test_mix.cpu().numpy(), columns=y_pred.columns).loc[ind,['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']], 
            yticklabels=[],  ax=axes[0,0], vmin=0, cmap=color)
axes[0,0].set_title('Ground Truth')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_w_Mixup_test_mix.csv', index_col=0).loc[ind,['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']],
            yticklabels=[], ax=axes[0,1], vmin=0,cmap=color)
axes[0,1].set_title('NN with Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_wo_Mixup_test_mix.csv', index_col=0).loc[ind,['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']], 
            yticklabels=[], ax=axes[0,2], vmin=0, cmap=color)
axes[0,2].set_title('NN without Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_ARIC_test_mix.csv', index_col=0).T.loc[[str(x) for x in ind],['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']], 
            yticklabels=[], ax=axes[0,3], vmin=0, cmap=color)
axes[0,3].set_title('ARIC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_CIBERSORT.csv', index_col=0).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']], 
            yticklabels=[], ax=axes[1,0], vmin=0, cmap=color)
axes[1,0].set_title('CBS')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv', index_col=0).loc[['X'+str(x) for x in ind],['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells']], 
            yticklabels=[], ax=axes[1,1], vmin=0, cmap=color)
axes[1,1].set_title('RPC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv', index_col=0).loc[['X'+str(x) for x in ind],['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells']], 
            yticklabels=[], ax=axes[1,2], vmin=0, cmap=color)
axes[1,2].set_title('CP')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NNLS_test_mix.csv', index_col=0).loc[ind,['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']], 
            yticklabels=[], ax=axes[1,3], vmin=0, cmap=color)
axes[1,3].set_title('NNLS')

plt.tight_layout()

plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/heatmap_test_mix_lymph.pdf', bbox_inches='tight')


# In[66]:


ind = pd.DataFrame(y_test_mix.cpu().numpy()).sort_values(by=[6,7,13,14], ascending=False).index

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

fig.suptitle('Mixed')

sns.heatmap(pd.DataFrame(y_test_mix.cpu().numpy(), columns=y_pred.columns).loc[ind,['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']], 
            yticklabels=[],  ax=axes[0,0], vmin=0, cmap=color)
axes[0,0].set_title('Ground Truth')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_w_Mixup_test_mix.csv', index_col=0
                        ).loc[ind,['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']],
            yticklabels=[], ax=axes[0,1], vmin=0,cmap=color)
axes[0,1].set_title('NN with Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_wo_Mixup_test_mix.csv', index_col=0
                        ).loc[ind,['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']], 
            yticklabels=[], ax=axes[0,2], vmin=0, cmap=color)
axes[0,2].set_title('NN without Mixup')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_ARIC_test_mix.csv', index_col=0
                        ).T.loc[[str(x) for x in ind],['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']], 
            yticklabels=[], ax=axes[0,3], vmin=0, cmap=color)
axes[0,3].set_title('ARIC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_CIBERSORT.csv', index_col=0
                        ).loc[ind, ['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']], 
            yticklabels=[], ax=axes[1,0], vmin=0, cmap=color)
axes[1,0].set_title('CBS')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv', index_col=0
                        ).loc[['X'+str(x) for x in ind],['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']], 
            yticklabels=[], ax=axes[1,1], vmin=0, cmap=color)
axes[1,1].set_title('RPC')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.CP.csv', index_col=0
                        ).loc[['X'+str(x) for x in ind],['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']], 
            yticklabels=[], ax=axes[1,2], vmin=0, cmap=color)
axes[1,2].set_title('CP')

sns.heatmap(pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NNLS_test_mix.csv', index_col=0
                        ).loc[ind,['Colon', 'Esophangus', 'SmallIntestine', 'Stomach']], 
            yticklabels=[], ax=axes[1,3], vmin=0, cmap=color)
axes[1,3].set_title('NNLS')

plt.tight_layout()

plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/heatmap_test_mix_gastro.pdf', bbox_inches='tight')


# In[ ]:





# In[143]:


df_a = pd.DataFrame(y_test_mix.cpu().numpy(), columns=y_pred.columns
                         ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_w_Mixup_test_mix.csv', index_col=0
                        ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]


# In[144]:


plt.scatter(np.array(df_a).flatten(), np.array(df_b).flatten(), alpha=0.2)


# In[145]:


plt.figure(figsize=(3,3))
sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1)
plt.ylim(-0.05,0.7)
plt.xticks([0,1,2], ['0', '0.2', '0.4'])
plt.title('NN with Mixup')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
sns.despine()
plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/violine_test_mix_lymph_NNmix.pdf', bbox_inches='tight')


# In[146]:


df_a = pd.DataFrame(y_test_mix.cpu().numpy(), columns=y_pred.columns
                         ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv', index_col=0
                        ).loc[['X'+str(x) for x in ind],  ['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells']]
plt.scatter(np.array(df_a).flatten(), np.array(df_b).flatten(), alpha=0.2)


# In[147]:


plt.figure(figsize=(3,3))
sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1)
plt.ylim(-0.05,0.7)
plt.xticks([0,1,2], ['0', '0.2', '0.4'])
plt.title('RPC')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
sns.despine()
plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/violine_test_mix_lymph_RPC.pdf', bbox_inches='tight')


# In[148]:


df_a = pd.DataFrame(y_test_mix.cpu().numpy(), columns=y_pred.columns
                         ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NNLS_test_mix.csv', index_col=0
                        ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]
plt.scatter(np.array(df_a).flatten(), np.array(df_b).flatten(), alpha=0.2)


# In[149]:


plt.figure(figsize=(3,3))
sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1)
plt.ylim(-0.05,0.7)
plt.xticks([0,1,2], ['0', '0.2', '0.4'])
plt.title('NNLS')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
sns.despine()
plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/violine_test_mix_lymph_NNLS.pdf', bbox_inches='tight')


# In[ ]:





# In[78]:


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
fig.suptitle('Lymphocytes')
df_a = pd.DataFrame(y_test_mix.cpu().numpy(), columns=y_pred.columns
                         ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_a).flatten()]).T, x=0, y=1, ax=axes[0,0])
axes[0,0].set_ylim(-0.05,0.7)
axes[0,0].set_xticklabels(['0', '0.2', '0.4'])
axes[0,0].set_title('Ground Truth')
axes[0,0].set_xlabel('Ground Truth')
axes[0,0].set_ylabel('Prediction')
sns.despine()

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_w_Mixup_test_mix.csv', index_col=0
                        ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1, ax=axes[0,1])
axes[0,1].set_ylim(-0.05,0.7)
axes[0,1].set_xticklabels(['0', '0.2', '0.4'])
axes[0,1].set_title('NN with Mixup')
axes[0,1].set_xlabel('Ground Truth')
axes[0,1].set_ylabel('Prediction')
sns.despine()

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NN_wo_Mixup_test_mix.csv', index_col=0
                        ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1, ax=axes[0,2])
axes[0,2].set_ylim(-0.05,0.7)
axes[0,2].set_xticklabels(['0', '0.2', '0.4'])
axes[0,2].set_title('NN without Mixup')
axes[0,2].set_xlabel('Ground Truth')
axes[0,2].set_ylabel('Prediction')
sns.despine()

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_ARIC_test_mix.csv', index_col=0
                        ).T.loc[[str(x) for x in ind], ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1, ax=axes[0,3])
axes[0,3].set_ylim(-0.05,0.7)
axes[0,3].set_xticklabels(['0', '0.2', '0.4'])
axes[0,3].set_title('ARIC')
axes[0,3].set_xlabel('Ground Truth')
axes[0,3].set_ylabel('Prediction')
sns.despine()

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_CIBERSORT.csv', index_col=0
                        ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1, ax=axes[1,0])
axes[1,0].set_ylim(-0.05,0.7)
axes[1,0].set_xticklabels(['0', '0.2', '0.4'])
axes[1,0].set_title('CBS')
axes[1,0].set_xlabel('Ground Truth')
axes[1,0].set_ylabel('Prediction')
sns.despine()


df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv', index_col=0
                        ).loc[['X'+str(x) for x in ind], ['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1, ax=axes[1,1])
axes[1,1].set_ylim(-0.05,0.7)
axes[1,1].set_xticklabels(['0', '0.2', '0.4'])
axes[1,1].set_title('RPC')
axes[1,1].set_xlabel('Ground Truth')
axes[1,1].set_ylabel('Prediction')
sns.despine()


df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.CP.csv', index_col=0
                        ).loc[['X'+str(x) for x in ind], ['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1, ax=axes[1,2])
axes[1,2].set_ylim(-0.05,0.7)
axes[1,2].set_xticklabels(['0', '0.2', '0.4'])
axes[1,2].set_title('CP')
axes[1,2].set_xlabel('Ground Truth')
axes[1,2].set_ylabel('Prediction')
sns.despine()

df_b = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/1/NNLS_test_mix.csv', index_col=0
                        ).loc[ind, ['B-cells', 'CD4+T-cells', 'CD8+T-cells', 'NK-cells']]

sns.violinplot(data=pd.DataFrame([np.array(df_a).flatten(), np.array(df_b).flatten()]).T, x=0, y=1, ax=axes[1,3])
axes[1,3].set_ylim(-0.05,0.7)
axes[1,3].set_xticklabels(['0', '0.2', '0.4'])
axes[1,3].set_title('NNLS')
axes[1,3].set_xlabel('Ground Truth')
axes[1,3].set_ylabel('Prediction')
sns.despine()

plt.tight_layout()

plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/violine_test_mix_lymph.pdf', bbox_inches='tight')


# In[ ]:





# In[ ]:





# ## stats

# In[3]:


import glob


# In[4]:


# list_res = glob.glob('../results/210107_compare_algorithms/*/results.csv')
list_res = glob.glob('../results/230531_compare_algorithms_test_indsamples_min3/*/results.csv')

list_df = []
for i,f in enumerate(list_res):
    d = pd.read_csv(f)
    d['seed'] = i
    list_df.append(d)

df_res = pd.concat(list_df)
list_order=['RF', 'SVR', 'GB', 'NNLS', 'NeuralNet_wo_Mixup', 'NeuralNet_w_Mixup']

fig, axes = plt.subplots(1,2, figsize=(6,4))

fig.suptitle('Purified')

sns.swarmplot(ax=axes[0], data=df_res, x="algorithm", y="loss", hue='algorithm', order=list_order)
axes[0].set_xticklabels(list_order, rotation=90)
axes[0].get_legend().remove()
axes[0].set_ylabel('MSE')
axes[0].set_ylim(0,0.06)
sns.despine()


sns.swarmplot(ax=axes[1], data=df_res, x="algorithm", y="test_time", hue='algorithm', order=list_order)
axes[1].set_yscale('log')
axes[1].set_xticklabels(list_order, rotation=90)
axes[1].get_legend().remove()
sns.despine()

plt.tight_layout()
plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/comp_time_mse.pdf' , bbox_inches='tight', transparent=True)


# In[6]:


df_res.groupby(by='algorithm').mean()


# In[152]:


fig, axes = plt.subplots(1,2, figsize=(6,4))
fig.suptitle('Mixed')

sns.swarmplot(ax=axes[0], data=df_res, x="algorithm", y="loss_mix", hue='algorithm', order=list_order)
axes[0].set_xticklabels(list_order, rotation=90)
axes[0].get_legend().remove()
axes[0].set_ylim(0,0.022)
axes[0].set_ylabel('MSE')
sns.despine()

sns.swarmplot(ax=axes[1], data=df_res, x="algorithm", y="test_time_mix", hue='algorithm', order=list_order)
axes[1].set_yscale('log')
axes[1].set_xticklabels(list_order, rotation=90)
axes[1].get_legend().remove()
sns.despine()

plt.tight_layout()
plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/comp_time_mix_mse.pdf' , bbox_inches='tight', transparent=True)


# In[113]:


plt.figure(figsize=(3,3))
sns.swarmplot(data=df_res, x="algorithm", y="training_time", hue='algorithm', order=list_order)
plt.yscale('log')
plt.xticks(range(len(list_order)),list_order, rotation=90)
plt.legend("")
sns.despine()

plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/comp_training_time.pdf' , bbox_inches='tight', transparent=True)


# In[130]:


d = pd.read_csv('../results/230531_compare_algorithms_test_indsamples_min3/results.tools_manuallyadded.csv')
d['seed'] = 0
d


# In[131]:


df_res = pd.concat([df_res[df_res['algorithm'] == 'NeuralNet_w_Mixup'], d])


# In[132]:


df_res


# In[133]:


df_res.algorithm.unique()


# In[135]:


list_order=['NeuralNet_w_Mixup', 'ARIC', 'CBS', 'RPC (EpiDISH)',
       'CBS (EpiDISH)', 'CP (EpiDISH)']

fig, axes = plt.subplots(1,2, figsize=(6,4))

fig.suptitle('Purified')

sns.swarmplot(ax=axes[0], data=df_res, x="algorithm", y="loss", hue='algorithm', order=list_order)
axes[0].set_xticklabels(list_order, rotation=90)
axes[0].get_legend().remove()
axes[0].set_ylim(0,0.038)
axes[0].set_ylabel('MSE')
sns.despine()


sns.swarmplot(ax=axes[1], data=df_res, x="algorithm", y="test_time", hue='algorithm', order=list_order)
axes[1].set_yscale('log')
axes[1].set_xticklabels(list_order, rotation=90)
axes[1].get_legend().remove()
sns.despine()

plt.tight_layout()
plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/comp_tools_time_mse.pdf' , bbox_inches='tight', transparent=True)


# In[136]:


fig, axes = plt.subplots(1,2, figsize=(6,4))
fig.suptitle('Mixed', fontsize=16)

sns.swarmplot(ax=axes[0], data=df_res, x="algorithm", y="loss_mix", hue='algorithm', order=list_order)
axes[0].set_xticklabels(list_order, rotation=90)
axes[0].get_legend().remove()
axes[0].set_ylim(0,0.0088)
axes[0].set_ylabel('MSE')
sns.despine()

sns.swarmplot(ax=axes[1], data=df_res, x="algorithm", y="test_time_mix", hue='algorithm', order=list_order)
axes[1].set_yscale('log')
axes[1].set_xticklabels(list_order, rotation=90)
axes[1].get_legend().remove()
sns.despine()

plt.tight_layout()
plt.savefig('../img/230531_compare_algorithms_test_indsamples_min3/comp_tools_time_mix_mse.pdf' , bbox_inches='tight', transparent=True)


# In[ ]:




