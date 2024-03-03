#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import yaml
from sklearn.impute import SimpleImputer

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(100)
get_ipython().run_line_magic('matplotlib', 'inline')
# %config InlineBackend.figure_format = 'retina'

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

pd.set_option('display.max_columns', 80)


# In[2]:


import numpy as np
import pandas as pd
import scanpy as sc


# In[3]:


import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# In[4]:


df = pd.read_csv('./cell_proportion_MinorGroup_allsamples.csv', index_col=0)  # your_data.csvを適切なファイルパスに置き換えてください


# In[5]:


# 比較したいサンプルペアのリストを作成する
sample_pairs = [('melanoma_180_S1', 'Fresh_180'),
                ('WGBS9-15_14_FFPE_180_S12', 'FFPE_180'),
                ('melanoma_183_S2', 'Fresh_183'),
                ('WGBS23-28_30_30_Fresh_183_S14', 'Fresh_183_2'),
                ('WGBS16-22_16_FFPE_183_S1', 'FFPE_183'),
                ('melanoma_184_S11', 'Fresh_184'),
                ('WGBS16-22_17_FFPE_184_S2', 'FFPE_184'),
                ('melanoma_185_S3', 'Fresh_185'),
                ('WGBS16-22_18_FFPE_185_S3', 'FFPE_185'),
                ('melanoma_186_low_S4', 'Fresh_186_low'),
                ('SCC_186_lower', 'Fresh_186_low_nano'),
                ('melanoma_186_up_S12', 'Fresh_186_up'),
                ('SCC_186_upper', 'Fresh_186_up_nano'),
                ('WGBS16-22_19_FFPE_186_S4', 'FFPE_186'),
                ('SCC_187_rep1', 'Fresh_187_nano1'),
                ('SCC_187_rep2', 'Fresh_187_nano2'),
                ('SCC_187_rep3', 'Fresh_187_nano3'),
                ('SCC_187_S16', 'Fresh_187'), 
                ('WGBS16-22_20_FFPE_187_S5', 'FFPE_187'),
                ('melanoma_191_S13', 'Fresh_191'),
                ('WGBS16-22_22_FFPE_191_S7', 'FFPE_191'), 
                ('melanoma_194_S14', 'Fresh_194'), 
                ('WGBS23-28_30_23_FFPE_194_S8', 'FFPE_194'),
                ('SCC_195_S7', 'Fresh_195'),
                ('WGBS23-28_30_24_FFPE_195_S9', 'FFPE_195'),
                ('SCC_196_S8', 'Fresh_196'),
                ('WGBS23-28_30_25_FFPE_196_S10', 'FFPE_196'),
                ('melanoma_200_infiltrate_S15', 'Fresh_200_inf'),
                ('melanoma_200_non-infiltrate_S5', 'Fresh_200_noinf'),
                ('melanoma_200_normal_S6', 'Fresh_200_normal'),
                ('WGBS23-28_30_27_FFPE_200_S12', 'FFPE_200'),
                ('EMPD_207_red_S9', 'Fresh_207_red'),
                ('EMPD_207_white_S10', 'Fresh_207_white'),
                ('WGBS23-28_30_28_FFPE_207_S13', 'FFPE_207')] 


# In[6]:


sample_pairs = [('melanoma_180', 'WGBS9-15_14_FFPE_180'),
                
#                ('melanoma_183', 'WGBS23-28_30_30_Fresh_183'),
                ('melanoma_183', 'WGBS16-22_16_FFPE_183'),
                
                ('melanoma_184', 'WGBS16-22_17_FFPE_184'),
#                ('melanoma_185', 'WGBS16-22_18_FFPE_185'),
                
                ('melanoma_186_low', 'WGBS16-22_19_FFPE_186'),
#                ('melanoma_186_up', 'WGBS16-22_19_FFPE_186'),
#                ('melanoma_186_up', 'WGBS16-22_19_FFPE_186'),
                
#                ('SCC187_rep1', 'WGBS16-22_20_FFPE_187'),
#                ('SCC187_rep2', 'WGBS16-22_20_FFPE_187'),
#                ('SCC187_rep3', 'WGBS16-22_20_FFPE_187'),
#                ('SCC_187', 'WGBS16-22_20_FFPE_187'), 
                
#                ('melanoma_191', 'WGBS16-22_22_FFPE_191'),
                ('melanoma_194', 'WGBS23-28_30_23_FFPE_194'), 
                ('SCC_195', 'WGBS23-28_30_24_FFPE_195'),
                
#                ('SCC_196', 'WGBS23-28_30_25_FFPE_196'),
                
#                ('melanoma_200_infiltrate', 'WGBS23-28_30_27_FFPE_200'),
#                ('melanoma_200_non-infiltrate', 'WGBS23-28_30_27_FFPE_200'),
#               ('melanoma_200_normal', 'WGBS23-28_30_27_FFPE_200'),
                
#                ('EMPD_207_red', 'WGBS23-28_30_28_FFPE_207'),
#                ('EMPD_207_white', 'WGBS23-28_30_28_FFPE_207')
               ]


# In[ ]:





# In[7]:


# pair
list_sample = ['melanoma_180_S1','WGBS9-15_14_FFPE_180_S12',
    'melanoma_183_S2', 'WGBS23-28_30_30_Fresh_183_S14', 'WGBS16-22_16_FFPE_183_S1',
    'melanoma_184_S11', 'WGBS16-22_17_FFPE_184_S2',
'melanoma_185_S3', 'WGBS16-22_18_FFPE_185_S3',
'melanoma_186_low_S4', 'SCC_186_lower', 'melanoma_186_up_S12', 'SCC_186_upper', 'WGBS16-22_19_FFPE_186_S4',
'SCC_187_rep1', 'SCC_187_rep2', 'SCC_187_rep3', 'SCC_187_S16','WGBS16-22_20_FFPE_187_S5',
'melanoma_191_S13', 'WGBS16-22_22_FFPE_191_S7',
'melanoma_194_S14', 'WGBS23-28_30_23_FFPE_194_S8',
'SCC_195_S7', 'WGBS23-28_30_24_FFPE_195_S9',
'SCC_196_S8', 'WGBS23-28_30_25_FFPE_196_S10',
'melanoma_200_infiltrate_S15', 'melanoma_200_non-infiltrate_S5', 'melanoma_200_normal_S6', 'WGBS23-28_30_27_FFPE_200_S12',
'EMPD_207_red_S9', 'EMPD_207_white_S10', 'WGBS23-28_30_28_FFPE_207_S13'
]
list_label = ['Fresh_180','FFPE_180',
    'Fresh_183','Fresh_183_2', 'FFPE_183', 
    'Fresh_184', 'FFPE_184',
    'Fresh_185', 'FFPE_185',
    'Fresh_186_low', 'Fresh_186_low_nano', 'Fresh_186_up', 'Fresh_186_up_nano', 'FFPE_186',
    'Fresh_187_nano1', 'Fresh_187_nano2', 'Fresh_187_nano3', 'Fresh_187', 'FFPE_187',
    'Fresh_191', 'FFPE_191', 
    'Fresh_194', 'FFPE_194', 
    'Fresh_195', 'FFPE_195', 
    'Fresh_196', 'FFPE_196',
    'Fresh_200_inf', 'Fresh_200_noinf', 'Fresh_200_normal', 'FFPE_200',
    'Fresh_207_red', 'Fresh_207_white', 'FFPE_207'
]


# In[8]:


# タプルのリストを作成
sample_label_pairs = list(zip(list_sample, list_label))

print(sample_label_pairs)


# In[9]:


from scipy.stats import pearsonr


# In[12]:


# データを整形してlong-formにする
long_form_data = []
for sample1, sample2 in sample_pairs:
    for index, row in df.iterrows():
        long_form_data.append({'cell_type': index,
                               'sample_pair': f'{sample1} vs {sample2}',
                               'x': row[sample1],
                               'y': row[sample2]})

long_form_df = pd.DataFrame(long_form_data)


# In[14]:


# データを100倍にする
long_form_df['x'] *= 100
long_form_df['y'] *= 100

# lmplotを使ってプロットするが、fit_regパラメータをFalseに設定して回帰曲線を表示しない
g = sns.lmplot(data=long_form_df, x='x', y='y', hue='sample_pair', fit_reg=False, height=6, aspect=1)

# 軸をログスケールに設定
plt.xscale('log')
plt.yscale('log')

# x軸とy軸のラベルを設定
plt.xlabel('Fresh (%)')
plt.ylabel('FFPE (%)')

# 全体のデータで回帰直線を計算
slope, intercept, r_value, p_value, std_err = linregress(np.log(long_form_df['x']), np.log(long_form_df['y']))

# 全体に対する対数スケールでの回帰直線をプロット
x_log = np.logspace(np.log10(min(long_form_df['x'])), np.log10(max(long_form_df['x'])), 100)
y_log = np.exp(intercept) * x_log**(slope)
plt.plot(x_log, y_log, linewidth=3, label='Overall Regression')

# R値とp-valueをプロットにテキストとして追加
plt.text(0.3, 0.9, f'R = {r_value:.2f}, P-value = {p_value:.2g}', ha='center', va='center', transform=plt.gca().transAxes)

# 凡例を追加
#plt.legend(title='Sample Pairs', bbox_to_anchor=(1.05, 1), loc='upper left')

# PDFとして保存
#plt.savefig("./lmplot_without_regression_percentage_pvalue.pdf", bbox_inches='tight')  # この環境では保存は行えないのでコメントアウト

# プロットを表示
plt.show()


# In[ ]:




