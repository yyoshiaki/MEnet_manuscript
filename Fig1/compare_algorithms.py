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

'''
python compare_algorithms.py ../params/230531_compare_algorithms_test_indsamples.yaml 0

input : yaml, seed

seq 10 | xargs -I{} python compare_algorithms.py ../params/230531_compare_algorithms_test_indsamples.yaml {}
'''


f = open(sys.argv[1], 'r+')
seed = int(sys.argv[2])
np.random.seed(seed)

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


list_results = []

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
    df_selected = \
        pd.read_csv(f_selected, index_col=0)
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
# y_test = torch.tensor(y_test, dtype=torch.long).to(device)

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
    
# data_set = Mixup_dataset(x_train, y_train, transform='mix', n_choise=5)
data_set = Mixup_dataset(x_train, y_train, transform='unmixed', n_choise=5)
dataloader = torch.utils.data.DataLoader(data_set, batch_size=100, shuffle=True)

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
    
    
class SmoothCrossEntropy(nn.Module):
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05,one_hotted=False):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.one_hotted = one_hotted
    def forward(self, x, target):
        if self.training:
            x = x.float()
            if self.one_hotted!=True:
                target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            if self.one_hotted!=True:
                loss = F.cross_entropy(x, target.long())
            else:
                loss = OneHotCrossEntropy(x, target)
            return loss
        
        
class MEnet(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout_rate, batch_norm, output_size):
        super(MEnet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size[0])
        self.fc2 = torch.nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.fc3 = torch.nn.Linear(self.hidden_size[1], self.output_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size[0])
        self.bn2 = nn.BatchNorm1d(self.hidden_size[1])
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout_rate[0])
        self.dropout2 = torch.nn.Dropout(dropout_rate[1])
        self.dropout3 = torch.nn.Dropout(dropout_rate[2])
            
            
    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


## create mix test
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

    x_test_mix.to_csv(f_pickle_test_mix)
    y_test_mix.to_csv(f_test_label_mix)

x_test_mix = torch.FloatTensor(np.array(x_test_mix.T)).to(device)
y_test_mix = torch.FloatTensor(np.array(y_test_mix)).to(device)

### NeuralNet without Mixup
print('NN without Mixup')
data_set = Mixup_dataset(x_train, y_train, transform='unmixed', n_choise=5)
# data_set = Mixup_dataset(x_train, y_train, transform='mix', n_choise=5)
dataloader = torch.utils.data.DataLoader(data_set, batch_size=100, shuffle=True)
# model = MEnet(x_test.shape[1], [2000, 2000], [0.1, 0.6, 0.6], True, labels_train.shape[1]).to(device)
model = MEnet(x_test.shape[1], [1000, 1000], [0, 0.1, 0.1], True, labels_train.shape[1]).to(device)

# criterion = torch.nn.CrossEntropyLoss()
# criterion = SmoothCrossEntropy(smoothing =0,one_hotted=True)
criterion = OneHotCrossEntropy()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)

model.eval()
y_pred = model(x_test)
before_train = criterion(y_pred, y_test)
print('Test loss before training' , before_train.item())

writer = SummaryWriter(log_dir=dict_input['output_dir'] + "/{}/NN_wo_Mixup".format(seed))

start = time.time()

epoch = 2000
list_loss = []
list_valloss = []
list_valloss_mix = []
for e in range(epoch):
    for i, (data, y_target) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, y_target)
        # loss = criterion(F.softmax(y_pred, dim=1), y_target)
        if i == 0:
            list_loss.append(loss.item())
        writer.add_scalar("Loss/train", loss, e)
        loss.backward()
        optimizer.step()
    
    if e % 100 == 0:
        print('Epoch {}: train loss: {}'.format(e, loss.item()))
        
    model.eval()

    with torch.no_grad():
        # print(x_test.shape)
        y_pred = model(x_test)
        valloss = criterion(y_pred, y_test) 
        # valloss = criterion(F.softmax(y_pred, dim=1), y_test)
        list_valloss.append(valloss.item())
        writer.add_scalar("Loss/validation", valloss, e)

        # print(x_test_mix.shape)
        y_pred = model(x_test_mix)
        valloss = criterion(y_pred, y_test_mix) 
        # valloss = criterion(F.softmax(y_pred, dim=1), y_test_mix)
        list_valloss_mix.append(valloss.item())

    # if e % 200 == 0:
    #     print('Epoch {}: train loss: {}, test loss: {}, test mix loss: {}'.format(e, loss.item(), list_valloss[-1], list_valloss_mix[-1]))

training_time = time.time() - start

model.eval()

start = time.time()
y_pred = model(x_test)
# after_train = criterion(y_pred, y_test) 
# after_train = min(list_valloss)
# after_train = list_valloss[-1]
test_time = time.time() - start
after_train = mean_squared_error(F.softmax(y_pred, dim=1).cpu().detach().numpy(), y_test.cpu().detach().numpy())

pd.DataFrame(F.softmax(y_pred, dim=1).cpu().detach().numpy(), columns=labels_train.columns).to_csv(dir_output + '/{}/NN_wo_Mixup_test.csv'.format(seed))


start = time.time()
y_pred = model(x_test_mix)
# after_train_mix = list_valloss_mix[-1]
test_time_mix = time.time() - start
after_train_mix = mean_squared_error(F.softmax(y_pred, dim=1).cpu().detach().numpy(), y_test_mix.cpu().detach().numpy())

pd.DataFrame(F.softmax(y_pred, dim=1).cpu().detach().numpy(), columns=labels_train.columns).to_csv(dir_output + '/{}/NN_wo_Mixup_test_mix.csv'.format(seed))

print('Test loss after Training' , after_train)
print('Mix Test loss after Training' , after_train_mix)
test_time = time.time() - start

# list_results.append(['NeuralNet_wo_Mixup', after_train.item(), training_time, test_time])
list_results.append(['NeuralNet_wo_Mixup', after_train, after_train_mix, training_time, test_time, test_time_mix])


### NeuralNet with Mixup
print('NN with Mixup')
# data_set = Mixup_dataset(x_train, y_train, transform='unmixed', n_choise=5)
data_set = Mixup_dataset(x_train, y_train, transform='mix', n_choise=15)
dataloader = torch.utils.data.DataLoader(data_set, batch_size=100, shuffle=True)
model = MEnet(x_test.shape[1], [1000, 1000], [0, 0.1, 0.1], True, labels_train.shape[1]).to(device)

# criterion = torch.nn.CrossEntropyLoss()
# criterion = SmoothCrossEntropy(smoothing =0,one_hotted=True)
criterion = OneHotCrossEntropy()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)

writer = SummaryWriter(log_dir=dict_input['output_dir'] + "/{}/NN_w_Mixup".format(seed))

start = time.time()

epoch = 200000
list_loss = []
list_valloss = []
list_valloss_mix = []
for e in range(epoch):
    for i, (data, y_target) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, y_target)
        # loss = criterion(F.softmax(y_pred, dim=1), y_target)
        if i == 0:
            list_loss.append(loss.item())
        writer.add_scalar("Loss/train", loss, e)
        loss.backward()
        optimizer.step()
    
    # if e % 1000 == 0:
    #     print('Epoch {}: train loss: {}'.format(e, loss.item()))
        
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test)
        valloss = criterion(y_pred, y_test) 
        # valloss = criterion(F.softmax(y_pred, dim=1), y_test)
        list_valloss.append(valloss.item())
        writer.add_scalar("Loss/validation", valloss, e)

        y_pred = model(x_test_mix)
        valloss = criterion(y_pred, y_test_mix) 
        # valloss = criterion(F.softmax(y_pred, dim=1), y_test_mix)
        list_valloss_mix.append(valloss.item())

        valrmse = mean_squared_error(F.softmax(y_pred, dim=1).cpu().detach().numpy(), y_test_mix.cpu().detach().numpy())
        
    if e % 500 == 0:
        print('Epoch {}: train loss: {}, test loss: {}, test mix loss: {}, test mix RMSE: {}'.format(e, loss.item(), list_valloss[-1], list_valloss_mix[-1], valrmse))
training_time = time.time() - start

model.eval()

start = time.time()
y_pred = model(x_test)
# after_train = criterion(y_pred, y_test) 
# after_train = min(list_valloss)
# after_train = list_valloss[-1]
test_time = time.time() - start
after_train = mean_squared_error(F.softmax(y_pred, dim=1).cpu().detach().numpy(), y_test.cpu().detach().numpy())

pd.DataFrame(F.softmax(y_pred, dim=1).cpu().detach().numpy(), columns=labels_train.columns).to_csv(dir_output + '/{}/NN_w_Mixup_test.csv'.format(seed))

start = time.time()
y_pred = model(x_test_mix)
# after_train_mix = list_valloss_mix[-1]
test_time_mix = time.time() - start
after_train_mix = mean_squared_error(F.softmax(y_pred, dim=1).cpu().detach().numpy(), y_test_mix.cpu().detach().numpy())

pd.DataFrame(F.softmax(y_pred, dim=1).cpu().detach().numpy(), columns=labels_train.columns).to_csv(dir_output + '/{}/NN_w_Mixup_test_mix.csv'.format(seed))

print('Test loss after Training' , after_train)
print('Mix Test loss after Training' , after_train_mix)


# list_results.append(['NeuralNet_w_Mixup', after_train.item(), training_time, test_time])
list_results.append(['NeuralNet_w_Mixup', after_train, after_train_mix, training_time, test_time, test_time_mix])

### NNLS
print('NNLS')
df_nnls_ref = pd.DataFrame(x_train.cpu().detach().numpy())
# df_nnls_ref.columns = df_train.index
df_nnls_ref['Tissue'] = \
    [labels_train.columns[np.where(x==1)[0][0]] for x in y_train.cpu().detach().numpy()]
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

# after_train = criterion(torch.FloatTensor(np.array(df_pred_nnls)).to(device), y_test) 
after_train = mean_squared_error(df_pred_nnls, y_test.cpu().detach().numpy())
pd.DataFrame(np.array(df_pred_nnls), columns=labels_train.columns).to_csv(dir_output + '/{}/NNLS_test.csv'.format(seed))
print('Test loss of nnls' , after_train.item())

arr_x_test_mix = x_test_mix.cpu().numpy()

start = time.time()
list_nnls = []
for i in range(arr_x_test_mix.shape[0]):
    list_nnls.append(nnls(preprocess(df_nnls_ref.T).T, arr_x_test_mix[i,:])[0])
test_time_mix = time.time() - start

df_pred_nnls = pd.DataFrame(list_nnls)
df_pred_nnls.columns = df_nnls_ref.columns
df_pred_nnls = df_pred_nnls.T.reindex(labels_train.columns).fillna(0).T
df_pred_nnls = df_pred_nnls.div(df_pred_nnls.sum(axis=1), axis=0)

# after_train_mix = criterion(torch.FloatTensor(np.array(df_pred_nnls)).to(device), y_test_mix) 
after_train_mix = mean_squared_error(df_pred_nnls, y_test_mix.cpu().detach().numpy())

pd.DataFrame(np.array(df_pred_nnls), columns=labels_train.columns).to_csv(dir_output + '/{}/NNLS_test_mix.csv'.format(seed))
print('Mix Test loss of nnls' , after_train_mix.item())

list_results.append(['NNLS', after_train.item(), after_train_mix.item(), 0, test_time, test_time_mix])

### SVR

start = time.time()
svr = RandomForestRegressor().fit(x_train.cpu().detach().numpy(),
                                y_train.cpu().detach().numpy())
training_time = time.time() - start

start = time.time()
preds = svr.predict(x_test.cpu().detach().numpy())
# after_train = criterion(torch.FloatTensor(preds).to(device), y_test)
test_time = time.time() - start
after_train = mean_squared_error(preds, y_test.cpu().detach().numpy())

pd.DataFrame(preds, columns=labels_train.columns).to_csv(dir_output + '/{}/SVR_test.csv'.format(seed))
print('Test loss of SVR' , after_train.item())

start = time.time()
preds = svr.predict(x_test_mix.cpu().detach().numpy())
# after_train_mix = criterion(torch.FloatTensor(preds).to(device), y_test_mix)
test_time_mix = time.time() - start
after_train_mix = mean_squared_error(preds, y_test_mix.cpu().detach().numpy())

pd.DataFrame(preds, columns=labels_train.columns).to_csv(dir_output + '/{}/SVR_test_mix.csv'.format(seed))
print('Mix Test loss of SVR' , after_train_mix.item())

list_results.append(['SVR', after_train.item(), after_train_mix.item(), training_time, test_time, test_time_mix])

### RF

start = time.time()
rf = RandomForestRegressor(random_state=seed).fit(x_train.cpu().detach().numpy(),
                                y_train.cpu().detach().numpy())
training_time = time.time() - start

start = time.time()
preds = rf.predict(x_test.cpu().detach().numpy())
# after_train = criterion(torch.FloatTensor(preds).to(device), y_test)
test_time = time.time() - start
after_train = mean_squared_error(preds, y_test.cpu().detach().numpy())

pd.DataFrame(preds, columns=labels_train.columns).to_csv(dir_output + '/{}/RF_test.csv'.format(seed))
print('Test loss of RF' , after_train.item())

start = time.time()
preds = rf.predict(x_test_mix.cpu().detach().numpy())
# after_train_mix = criterion(torch.FloatTensor(preds).to(device), y_test_mix)
test_time_mix = time.time() - start
after_train_mix = mean_squared_error(preds, y_test_mix.cpu().detach().numpy())

pd.DataFrame(preds, columns=labels_train.columns).to_csv(dir_output + '/{}/RF_test_mix.csv'.format(seed))
print('Mix Test loss of RF' , after_train_mix.item())

list_results.append(['RF', after_train.item(), after_train_mix.item(), training_time, test_time, test_time_mix])

### lightGBM

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

params={'learning_rate': 0.5,
        'objective':'mae', 
        'metric':'mae',
        'num_leaves': 9,
        'verbose': -1,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7,
        'seed' : seed
       }
reg = MultiOutputRegressor(lgb.LGBMRegressor(**params, n_estimators=500))

start = time.time()
reg.fit(x_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
training_time = time.time() - start

start = time.time()
preds = reg.predict(x_test.cpu().detach().numpy())
test_time = time.time() - start
# after_train = criterion(torch.FloatTensor(preds).to(device), y_test) 
after_train = mean_squared_error(preds, y_test.cpu().detach().numpy())
pd.DataFrame(preds, columns=labels_train.columns).to_csv(dir_output + '/{}/GB_test.csv'.format(seed))
print('Test loss of Gradient Boosting' , after_train.item())

start = time.time()
preds = reg.predict(x_test_mix.cpu().detach().numpy())
test_time_mix = time.time() - start
# after_train_mix = criterion(torch.FloatTensor(preds).to(device), y_test_mix) 
after_train_mix = mean_squared_error(preds, y_test_mix.cpu().detach().numpy())
pd.DataFrame(preds, columns=labels_train.columns).to_csv(dir_output + '/{}/GB_test_mix.csv'.format(seed))
print('Mix Test loss of Gradient Boosting' , after_train_mix.item())

list_results.append(['GB', after_train.item(), after_train_mix.item(), training_time, test_time, test_time_mix])

### save result
pd.DataFrame(list_results, columns = ['algorithm', 'loss', 'loss_mix', 'training_time', 'test_time', 'test_time_mix']
            ).to_csv(dir_output + '/{}/results.csv'.format(seed), index=None)