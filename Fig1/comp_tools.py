import time
import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.loss import _WeightedLoss
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from ARIC import *

f = open(sys.argv[1], 'r+')
# seed = int(sys.argv[2])
# np.random.seed(seed)

dict_input = yaml.load(f)
# os.makedirs(dict_input['output_dir'] + "/{}".format(seed), exist_ok=True)

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

criterion = OneHotCrossEntropy()

list_results = []

## ARIC
start = time.time()
ARIC(mix_path=f'{dir_output}/x_test_nnls.csv', 
     ref_path=f'{dir_output}/ref_nnls.csv',
     is_methylation=False,
     save_path=f'{dir_output}/pred_ARIC_test.csv')
test_time = time.time() - start

after_train = criterion(torch.FloatTensor(np.array(pd.read_csv(f'{dir_output}/pred_ARIC_test.csv', index_col=0).T)).to(device), 
                        y_test)
print('Test loss of ARIC' , after_train.item())

start = time.time()
ARIC(mix_path=f'{dir_output}/x_test_mix_nnls.csv', 
     ref_path=f'{dir_output}/ref_nnls.csv',
     is_methylation=False,
     save_path=f'{dir_output}/pred_ARIC_test_mix.csv')
test_time_mix = time.time() - start

after_train_mix = criterion(torch.FloatTensor(np.array(pd.read_csv(f'{dir_output}/pred_ARIC_test_mix.csv', index_col=0).T)).to(device), 
                        y_test_mix)
print('Mix Test loss of ARIC' , after_train.item())
list_results.append(['ARIC', after_train.item(), after_train_mix.item(), 0, test_time, test_time_mix])


### save result
pd.DataFrame(list_results, columns = ['algorithm', 'loss', 'loss_mix', 'training_time', 'test_time', 'test_time_mix']
            ).to_csv(dir_output + '/results.tools.csv', index=None)