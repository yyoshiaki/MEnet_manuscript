#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pickle
import os
import yaml
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

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


# In[22]:


import pandas as pd
import os

# フォルダのリスト
folders = ['ACC_MEnet', 'BLCA_MEnet', 'BRCA_MEnet', "CESC_MEnet","GBM_MEnet",
           "HNSC_MEnet","KICH_MEnet","KIRC_MEnet","KIRP_MEnet","LGG_MEnet","LIHC_MEnet",
           "LUAD_MEnet","LUSC_MEnet","PRAD_MEnet","READ_MEnet","SKCM_MEnet","THCA_MEnet"]  # フォルダ名を適切に指定してください

# 結合後のデータフレームを初期化
merged_df = pd.DataFrame()

# 各フォルダ内のCSVファイルを読み込み、結合
for folder in folders:
    csv_files = [file for file in os.listdir(folder) if file.endswith('_MajorGroup.csv')]
    
    for file in csv_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path, delimiter=',', index_col=0)
        merged_df = pd.concat([merged_df, df], axis=1)

# 結果を表示
merged_df


# In[23]:


merged_df.loc['Cancer'] = merged_df.loc['Adipocytes'] + merged_df.loc['AdrenalGland'] + merged_df.loc['Nerve']+ merged_df.loc['Skin']+ merged_df.loc['Muscle']+ merged_df.loc['Cardiovascular']+ merged_df.loc['Esophangus']+ merged_df.loc['Stomach']+ merged_df.loc['SmallIntestine']+merged_df.loc['Colon']+ merged_df.loc['Liver_Pancreas']+ merged_df.loc['Thyroid']+ merged_df.loc['Ovary']+ merged_df.loc['Testis']+ merged_df.loc['Kidney'] + merged_df.loc['Bladder']+ merged_df.loc['Prostate']+ merged_df.loc['Breast']+ merged_df.loc['Lung'] + merged_df.loc['Fibroblast']+ merged_df.loc['Endothelial-cells']
merged_df.loc['Immune'] = merged_df.loc['Meg-Ery'] + merged_df.loc['Neutrophils'] + merged_df.loc['Eosinophils']+ merged_df.loc['Myeloid']+ merged_df.loc['B-cells']+ merged_df.loc['CD4+T-cells']+ merged_df.loc['CD8+T-cells']+ merged_df.loc['NK-cells']


# In[24]:


merged_df_2 = merged_df.loc[['Immune', 'Cancer']]

merged_df_2


# In[25]:


EST = pd.read_excel('./41467_2015_BFncomms9971_MOESM1236_ESM.xlsx', index_col='Sample ID')
EST


# In[26]:


merged_df_3 = merged_df_2.T
merged_df_3


# In[27]:


merged_data = pd.merge(merged_df_3, EST, left_index=True, right_index=True)
merged_data


# In[28]:


# ユニークなCancertypeのリストを取得
cancer_types = merged_data['Cancer type'].unique()


# In[29]:


# 結果を格納するデータフレームを初期化
result_df = pd.DataFrame(columns=['Cancer type', 'Correlation', 'Variable1', 'Variable2', 'P-value', 'R-value', 'N'])


# In[30]:


# 各Cancertypeについて相関を計算
for cancer_type in cancer_types:
    # 現在のCancertypeのデータを抽出
    subset_df = merged_data[merged_data['Cancer type'] == cancer_type]
    
    # 相関を計算する列のリスト
    columns_to_calculate = [['Immune', 'Immune(LUMP)'],
                            ['Cancer', 'LUMP'],['Cancer', 'ESTIMATE'], ['Cancer', 'ABSOLUTE'],['Cancer', 'IHC'], ['Cancer', 'CPE']]
     
    for col1, col2 in columns_to_calculate:
        valid_mask = subset_df[[col1, col2]].notnull().all(axis=1)
        valid_data = subset_df.loc[valid_mask, [col1, col2]]
        
        if valid_data.shape[0] > 0:
            corr, p_value = pearsonr(valid_data[col1].values.flatten(), valid_data[col2].values.flatten())
            n = valid_data.shape[0]
            result_df = pd.concat([result_df, pd.DataFrame({'Cancer type': [cancer_type], 'Correlation': [f'{col1}-{col2}'],
                                                            'Variable1': [col1], 'Variable2': [col2],
                                                            'P-value': [p_value], 'R-value': [corr], 'N': [n]})], ignore_index=True)
        else:
            result_df = pd.concat([result_df, pd.DataFrame({'Cancer type': [cancer_type], 'Correlation': [f'{col1}-{col2}'],
                                                            'Variable1': [col1], 'Variable2': [col2],
                                                            'P-value': [float('nan')], 'R-value': [float('nan')], 'N': [0]})], ignore_index=True)


# In[31]:


result_df


# In[32]:


immune = result_df[result_df["Correlation"] == "Immune-Immune(LUMP)"]
immune


# In[33]:


cancer = result_df[result_df["Correlation"] != "Immune-Immune(LUMP)"]
cancer


# In[34]:


# ピボットテーブルを作成
pivot_table = cancer.pivot_table(index='Variable2', columns='Cancer type', values='R-value')

# P-valueを*に変換する関数
def pvalue_to_stars(pvalue):
    if pd.isnull(pvalue):
        return ''
    elif pvalue < 1e-10:
        return '***'
    elif pvalue < 1e-5:
        return '**'
    elif pvalue < 0.05:
        return '*'
    else:
        return ''

# P-valueを*に変換したデータフレームを作成
pvalue_df = cancer.pivot_table(index='Variable2', columns='Cancer type', values='P-value')
pvalue_df = pvalue_df.applymap(pvalue_to_stars)

# ヒートマップを作成
fig, ax = plt.subplots(figsize=(10, 10))
heatmap = sns.heatmap(pivot_table, cmap='coolwarm', linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_aspect('equal')

# セルに星印を追加
for i, row in enumerate(pvalue_df.itertuples(index=False)):
    for j, val in enumerate(row):
        if val:  # 星印があれば黒色で表示
            text = ax.text(j + 0.5, i + 0.5, val, ha="center", va="center", color='k')

# ラベルを設定
ax.set_title('Correlation to tumor purity scores')
ax.set_xlabel('')
plt.subplots_adjust(left=0.2)
plt.yticks(rotation=0)

# カラースケールを横向きに配置
cbar = ax.collections[0].colorbar
cbar.ax.set_position([0.15, 0.05, 0.7, 0.03])

plt.tight_layout()
plt.show()


# In[35]:


# ピボットテーブルを作成
pivot_table = immune.pivot_table(index='Variable2', columns='Cancer type', values='R-value')

# P-valueを*に変換する関数
def pvalue_to_stars(pvalue):
    if pd.isnull(pvalue):
        return 'NA'
    elif pvalue < 1e-10:
        return '***'
    elif pvalue < 1e-5:
        return '**'
    elif pvalue < 0.05:
        return '*'
    else:
        return ''

# P-valueを*に変換したデータフレームを作成
pvalue_df = immune.pivot_table(index='Variable2', columns='Cancer type', values='P-value')
pvalue_df = pvalue_df.applymap(pvalue_to_stars)

# ヒートマップを作成
fig, ax = plt.subplots(figsize=(10, 10))  # 正方形にするため、同じサイズを指定
sns.heatmap(pivot_table, cmap='coolwarm', linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_aspect('equal')  # セルを正方形に

# セルに星印を追加
for i, row in enumerate(pvalue_df.itertuples(index=False)):
    for j, val in enumerate(row):
        if val:  # 星印があれば黒色で表示
            text = ax.text(j + 0.5, i + 0.5, val, ha="center", va="center", color='k')

ax.invert_yaxis()

# ラベルを設定
ax.set_title('Correlation to tumor purity scores')
ax.set_xlabel('')
plt.subplots_adjust(left=0.2)  # Y軸ラベルの横向き表示のためにleftを調整
plt.yticks(rotation=0)  # Y軸ラベルを横向きに


# カラースケールを横向きに
cax = plt.gcf().axes[-1]
cax.set_position([0.15, 0.05, 0.65, 0.03])  # カラースケールの位置を調整

plt.tight_layout()
plt.show()


# In[ ]:




