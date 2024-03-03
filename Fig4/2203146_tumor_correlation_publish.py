#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from numpy.random import randn
import pandas as pd
import math

from scipy import stats

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import pingouin as pg


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import math

#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# In[3]:


#read data
df_FCM = pd.read_excel("../生データ/211227_大倉研_共有.xlsx",usecols=
                       [1,2,3,4,8,9,10,11,12,13,14,15,
                        16,17,18,19,20,21,22,23,24,25,26],header=2)
df_MEnet = pd.read_csv("../生データ/cell_proportion_MinorGroup_allsamples 2.csv",index_col=0)
df_MEnet *= 100
df_ID = pd.read_csv('../生データ/220108_MEnet_FACS_IDs 2.tsv',encoding="shift-jis", sep='\t', usecols=[1,2,3,4,5])


# In[4]:


df_FCM = pd.merge(df_FCM, df_ID, on=['ID', '癌種', '組織', '特記事項']).dropna(subset=['MEnetID'])
df_FCM = df_FCM.dropna(subset=['% Live CD45+']).fillna(0)
df_FCM.head()


# In[5]:


# multiply CD45% to all cells in df_FCM
l_cd45 = ['Lym', 'Mye', 'Neu',
       'Mφ', 'pDC', 'cDC1', 'cDC2', 'CD4Tconv', 'Treg', 'CD8T', 'NK', 'CD4Tn',
       'eTreg', 'nTreg', 'CD4Teff', 'CD8Tn', 'CD8Teff', 'CD8Tex']
for c in l_cd45:
    df_FCM[c] = df_FCM[c] * df_FCM['% Live CD45+'] / 100
df_FCM.head()


# In[6]:


df_MEnet = df_MEnet[df_FCM['MEnetID']].T
df_MEnet


# In[7]:


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
                      "Mac" : ['Macrophage'],
                      "DC" : ['Dendritic-cells'],
                      "naive Tconv" : ["naive Tconv"],
                      "memory Tconv" : ["memory Tconv"],
                      "naive Treg" : ['naive Treg'],
                      "activated Treg" : ['activated Treg'],
                      "naive CTL" : ['naive CTL'],
                      "memory CTL" : ['memory CTL'],
                      "exausted CTL" : ['exausted CTL'],
                     }

dict_corresp_FCM = {"CD45" : ["% Live CD45+"],
                    'Mye' : ['Mye'],
                    'Lym': ['Lym'],
                    'CD4T' : ['CD4Tconv', 'Treg'],
                    "CD4Tconv" : ['CD4Tconv'],
                    "Treg" : ['Treg'],
                    "CD8T" : ['CD8T'],
                    "NK" : ['NK'],
                    "Neu" : ['Neu'],
                    "Mac" : ['Mφ'],
                    "DC" : ['pDC', 'cDC1', 'cDC2'],
                    "naive Tconv" : ['CD4Tn'],
                    "memory Tconv" : ['CD4Teff'],
                    "naive Treg" : ['nTreg'],
                    "activated Treg" : ['eTreg'],
                    "naive CTL" : ['CD8Tn'],
                    "memory CTL" : ['CD8Teff'],
                    "exausted CTL" : ['CD8Tex'],

                   }

assert dict_corresp_MEnet.keys() == dict_corresp_FCM.keys()


# In[10]:


list_value = []
for k in dict_corresp_MEnet.keys():
    for i1, i2 in zip(df_FCM.index, df_MEnet.index):
        assert df_FCM.loc[i1, 'MEnetID'] == i2
        list_value.append([i2, df_FCM.loc[i1, '癌種'], df_FCM.loc[i1, '組織'], df_FCM.loc[i1, '特記事項'], 'FCM', k, 
                           df_FCM.loc[i1, dict_corresp_FCM[k]].sum(), df_MEnet.loc[i2, dict_corresp_MEnet[k]].sum()])

df = pd.DataFrame(list_value, columns=['MEnetID', 'cancer_type', 'tissue', 'note', 'assay', 'cell_type', 'FCM', 'MEnet'])
df


# In[13]:


unique_cell_types = df['cell_type'].unique()

for cell_type in unique_cell_types:
    subset_df = df[df['cell_type'] == cell_type]

    g = sns.lmplot(x='MEnet', y='FCM', data=subset_df, sharey=False, sharex=False,
                   scatter_kws={'s': 100}, 
                   line_kws={'lw': 4}) 

    slope, intercept, r_value, p_value, std_err = linregress(subset_df['FCM'], subset_df['MEnet'])

    plt.text(0.3, 0.9, f'R = {r_value:.2f}, P-value = {p_value:.3g}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title(cell_type)
    
    #g.savefig(f'./230922_tumor_corr/plot_{cell_type}.pdf')


# In[ ]:





# In[ ]:




