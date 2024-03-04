#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(100)
get_ipython().run_line_magic('matplotlib', 'inline')
# %config InlineBackend.figure_format = 'retina'
import pingouin as pg

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

pd.set_option('display.max_columns', 80)


# In[2]:


df_samples = pd.read_csv('GSE167998_series_matrix.txt.gz', sep='\t', skiprows=37, index_col=0)
df_samples.index = df_samples.index.str.split('!Sample_').str.get(1)
df_samples.loc['sampleid'] = df_samples.loc['supplementary_file'].iloc[0].str.extract('GSM\d*_(\d.*)_Grn', expand=False)
df_samples.head()


# In[6]:


list_label = ['mix', 'mix', 'mix', 'mix', 'mix',
       'mix', 'mix', 'mix', 'mix', 'mix', 'mix',
       'mix', 'cd4nv', 'cd4nv', 'cd4mem', 'cd4nv',
       'baso', 'cd4mem', 'cd4nv', 'cd4mem', 'cd4nv',
       'baso', 'baso', 'cd4mem', 'bmem', 'bnv', 'bnv',
       'bmem', 'bmem', 'bnv', 'bmem', 'bnv', 'bmem',
       'baso', 'bmem', 'treg', 'treg', 'treg', 'cd8t',
       'cd8t', 'cd8t', 'cd8t', 'cd8t', 'cd8t', 'cd8t', 'cd8t',
       'cd8t', 'eos', 'eos', 'eos', 'nk', 'neu',
       'nk', 'mono', 'nk', 'WB', 'baso', 'baso',
       'eos', 'mono', 'mono', 'mono', 'WB', 'mono',
       'neu', 'WB', 'nk', 'PCA']


# In[7]:


list_celltype = ['treg', 'cd4nv', 'cd4mem', 'cd8t',
                'bnv', 'bmem', 'nk',
                'mono', 'neu', 'eos']
df_answer = df_samples.loc['characteristics_ch1']
df_answer.columns = df_samples.loc['sampleid']
df_answer.index = df_answer.iloc[:,0].str.split(': ').str.get(0)
df_answer = df_answer.loc[list_celltype]
for c,l in zip(df_answer.columns, list_label):
    df_answer[c] = df_answer[c].str.split(': ').str.get(1).str.replace("NA", '0').astype(float)
    if l in list_celltype:
        df_answer[c] = [1 if c == l else 0 for c in list_celltype]
df_answer = df_answer.loc[:,df_answer.sum(axis=0) > 0]
df_answer = df_answer / df_answer.sum()
df_answer.head()


# In[10]:


df_res = pd.read_csv('./231107_merge_cell_proportion_MinorGroup.csv', index_col=0)
df_res.head()


# In[15]:


dict_corresp_MEnet = {"CD45" : ['Meg-Ery', 'Neutrophils', 'Eosinophils',
                                'Monocytes', 'Macrophage', 'Dendritic-cells', 'naive B-cells',
                                'mature B-cells', 'naive Treg', 'activated Treg', 'naive Tconv',
                                'memory Tconv', 'naive CTL', 'memory CTL', 'exausted CTL', 'NK-cells'],
                      'Mye' : ['Meg-Ery', 'Neutrophils', 'Eosinophils',
                               'Monocytes', 'Macrophage', 'Dendritic-cells'],
                      'Lym' : ['naive B-cells', 'mature B-cells', 'naive Treg', 'activated Treg', 
                               'naive Tconv', 'memory Tconv', 'naive CTL', 'memory CTL', 'exausted CTL', 'NK-cells'],
                      'CD4T' : ['naive Treg', 'activated Treg', 'naive Tconv', 'memory Tconv'],
                      "CD4Tconv" : ['naive Tconv', 'memory Tconv'],
                      "Treg" : ['naive Treg', 'activated Treg'],
                      "CD8T" : ['naive CTL', 'memory CTL', 'exausted CTL'],
                      "NK" : ['NK-cells'],
                      "Neu" : ['Neutrophils'],
                      "Eos" : ["Eosinophils"],
                      "Monocytes" : ["Monocytes"],
                      "naive Tconv" : ["naive Tconv"],
                      "memory Tconv" : ["memory Tconv"],
                      "naive B-cells" : ["naive B-cells"],
                      "mature B-cells" : ["mature B-cells"]
                     } 

dict_corresp_FCM = {"CD45" : ['treg', 'cd4nv', 'cd4mem', 'cd8t', 'bnv', 'bmem', 'nk', 'mono', 'neu', 'eos'],
                    'Mye' : ['mono', 'neu', 'eos'],
                    'Lym': ['treg', 'cd4nv', 'cd4mem', 'cd8t', 'bnv', 'bmem', 'nk'],
                    'CD4T' : ['treg', 'cd4nv', 'cd4mem'],
                    "CD4Tconv" : ['cd4nv', 'cd4mem'],
                    "Treg" : ['treg'],
                    "CD8T" : ['cd8t'],
                    "NK" : ['nk'],
                    "Neu" : ['neu'],
                    "Eos" : ["eos"],
                    "Monocytes" : ["mono"],
                    "naive Tconv" : ['cd4nv'],
                    "memory Tconv" : ['cd4mem'],
                    "naive B-cells" : ['bnv'],
                    "mature B-cells" : ['bmem']
                   }

assert dict_corresp_MEnet.keys() == dict_corresp_FCM.keys()


# In[17]:


df_FCM = df_answer.T
df_MEnet = df_res.T

df_FCM = df_FCM.loc[list(set(df_FCM.index) & set(df_MEnet.index))]
df_MEnet = df_MEnet.loc[list(set(df_FCM.index) & set(df_MEnet.index))]

list_value = []
for k in dict_corresp_MEnet.keys():
    for i1, i2 in zip(df_FCM.index, df_MEnet.index):
        # print(k,i1,i2)
        list_value.append([i2, k, df_FCM.loc[i1, dict_corresp_FCM[k]].sum(), df_MEnet.loc[i2, dict_corresp_MEnet[k]].sum()])

df = pd.DataFrame(list_value, columns=['MEnetID', 'cell_type', 'FCM', 'MEnet'])
df.head()


# In[28]:


list_ds = []

for c in dict_corresp_FCM.keys():
    print(c)
    data_subset = df[df['cell_type']==c].copy()  # データのコピーを作成して操作することで、元のデータを保持

    # データを100倍にする
    data_subset['MEnet'] *= 100
    data_subset['FCM'] *= 100

    lm = sns.lmplot(data=data_subset, x='MEnet', y='FCM',sharey=False, sharex=False,
                   scatter_kws={'s': 100}, # これでドットのサイズを変更
                   line_kws={'lw': 4})
    plt.title(c)

    # 相関係数(R値)とp-valueを計算
    slope, intercept, r_value, p_value, std_err = linregress(data_subset['MEnet'], data_subset['FCM'])

    # R値とp-valueをプロットにテキストとして追加
    plt.text(0.5, 0.9, f'R = {r_value:.2f}, p-value = {p_value:.2e}', ha='center', va='center', transform=plt.gca().transAxes)
    
    if c != "CD45":
        d_s = pg.pairwise_corr(data_subset, method='pearson')
        d_s['cell_type'] = c
        list_ds.append(d_s)
    
    # 各プロットをPDFファイルとして保存
    plt.savefig(f'plot_{c}_percentage_p-value_1927.pdf')
    
df_stats = pd.concat(list_ds)
df_stats


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




