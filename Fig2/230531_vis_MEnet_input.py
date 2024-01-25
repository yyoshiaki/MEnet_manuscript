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


# ## load input

# In[2]:


version = '230228'
f_pickle = '../MEnet/{}/df.pickle'.format(version)
f_yaml = '../MEnet/{}/MEnet_train.yaml'.format(version)

out_df = '../MEnet/{}/MEnet_ref.csv'.format(version)
out_dir_img = '../MEnet/{}/vis'.format(version)

os.makedirs(out_dir_img, exist_ok=True)


# In[3]:


with open(f_yaml, 'r') as f:
    dict_input = yaml.load(f)


# In[4]:


dict_input


# In[5]:


with open(f_pickle, 'rb') as f:
    df_input = pickle.load(f)
df_input.to_csv(out_df)


# In[6]:


df_input


# In[7]:


df_ref = pd.read_csv(dict_input['ref_table'], index_col=0)
df_ref = df_ref.loc[df_input.columns]
df_ref.head()


# In[36]:


plt.figure(figsize=(6,2))
my_colors = [(x/40.0, x/80.0, .9)
        for x in range(len(df_ref['MinorGroup'].unique()))]
df_ref['MinorGroup'].value_counts().plot.bar(color=my_colors)
plt.yscale('log')
plt.ylim(1,df_ref['MinorGroup'].value_counts().max())
sns.despine()
plt.savefig('{}/bar_ref_minor.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[35]:


plt.figure(figsize=(6,2))
my_colors = [(x/40.0, x/80.0, .9)
        for x in range(len(df_ref['Tissue'].unique()))]
df_ref['Tissue'].value_counts().plot.bar(color=my_colors)
plt.yscale('log')
plt.ylim(1,df_ref['Tissue'].value_counts().max())
sns.despine()
plt.savefig('{}/bar_ref_tissue.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[34]:


plt.figure(figsize=(3,2))
dict_colors = {'WGBS' : 'red', 'RRBS' : 'orange', '450K' : 'green', 'EPIC' : 'lime', 'Nanopore' : 'blue'}
df_ref['Assay'].value_counts()[dict_colors.keys()].plot.bar(color=dict_colors.values())
plt.yscale('log')
plt.ylim(1,df_ref['Assay'].value_counts().max())
sns.despine()
plt.savefig('{}/bar_ref_assay.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[11]:


df_ref


# In[13]:


df_ref[df_ref.Tissue.isna()]


# In[15]:


df_ref.Tissue.value_counts().shape


# In[16]:


df_ref.MinorGroup.value_counts().shape


# In[17]:


imp = SimpleImputer(missing_values=np.nan, strategy=dict_input['fill'])
arr_input_imp = imp.fit_transform(df_input.T).T


# In[18]:


df_input


# In[19]:


df_input.isna().sum(axis=1).sort_values(ascending=False)


# In[20]:


pd.DataFrame(arr_input_imp, columns=df_ref['MinorGroup'], index=df_input.index).loc[(df_input.isna().sum(axis=1) < 634 * 0.4)]


# In[21]:


list(df_input.isna().sum(axis=1) < 634 * 0.4)


# In[22]:


df_ref['Assay'].unique()


# In[23]:


df_ref.shape[0]


# In[24]:


# plt.figure(figsize=(30,10))
sns.clustermap(pd.DataFrame(arr_input_imp, columns=df_ref['MinorGroup']), cmap='cividis', yticklabels=False,
            col_cluster=False, row_cluster=False, col_colors=[dict_colors[x] for x in  df_ref['Assay']],
            figsize=(30,10))
plt.savefig('{}/heatmap_ref.png'.format(out_dir_img), 
            bbox_inches='tight', dpi=300)


# In[25]:


# plt.figure(figsize=(30,10))
sns.clustermap(pd.DataFrame(arr_input_imp, columns=df_ref['MinorGroup'], index=df_input.index).loc[(df_input.isna().sum(axis=1) < df_ref.shape[0] * 0.4)], 
            cmap='cividis', yticklabels=False,
            col_cluster=False, row_cluster=False, col_colors=[dict_colors[x] for x in  df_ref['Assay']],
            figsize=(30,10))
plt.savefig('{}/heatmap_ref_nan_under40.png'.format(out_dir_img), 
            bbox_inches='tight', dpi=300)


# In[26]:


# plt.figure(figsize=(30,10))
sns.clustermap(pd.DataFrame(arr_input_imp, columns=df_ref['MinorGroup'], index=df_input.index).loc[(df_input.isna().sum(axis=1) < df_ref.shape[0] * 0.2)], 
            cmap='cividis', yticklabels=False,
            col_cluster=False, row_cluster=False, col_colors=[dict_colors[x] for x in  df_ref['Assay']],
            figsize=(30,10))
plt.savefig('{}/heatmap_ref_nan_under20.png'.format(out_dir_img), 
            bbox_inches='tight', dpi=300)


# In[27]:


# plt.figure(figsize=(30,10))
sns.clustermap(pd.DataFrame(arr_input_imp, columns=df_ref['MinorGroup'], index=df_input.index).loc[(df_input.isna().sum(axis=1) < df_ref.shape[0] * 0.1)], 
            cmap='cividis', yticklabels=False,
            col_cluster=False, row_cluster=False, col_colors=[dict_colors[x] for x in  df_ref['Assay']],
            figsize=(30,10))
plt.savefig('{}/heatmap_ref_nan_under10.png'.format(out_dir_img), 
            bbox_inches='tight', dpi=300)


# In[ ]:





# In[ ]:





# In[28]:


dict_shape = {'WGBS' : 'o', 'RRBS' : 's', 'Nanopore' : '^', '450K' : 'x', 'EPIC' : '+'}


# In[39]:


arr_input_imp.T.shape


# In[ ]:





# In[52]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

embedding = umap.UMAP(metric='cosine', n_neighbors=200, min_dist=0.8, n_components=2
                      ).fit_transform(
                          PCA(n_components=200).fit_transform(
                              StandardScaler().fit_transform(arr_input_imp.T)))

df_ref['umap_x'] = embedding[:,0]
df_ref['umap_y'] = embedding[:,1]

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_ref, x='umap_x', y='umap_y', hue='Tissue', palette="tab20", style='Assay')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
sns.despine()
plt.savefig('{}/umap_ref_tissue.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[54]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_ref, x='umap_x', y='umap_y', hue='Tissue', palette="tab20", style='Assay')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
sns.despine()
plt.savefig('{}/umap_ref_tissue.png'.format(out_dir_img), 
            bbox_inches='tight')


# In[55]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_ref, x='umap_x', y='umap_y', hue='MinorGroup', palette="tab20", style='Assay')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
sns.despine()
plt.savefig('{}/umap_ref_minor.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[56]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_ref, x='umap_x', y='umap_y', hue='MinorGroup', palette="tab20", style='Assay')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
sns.despine()
plt.savefig('{}/umap_ref_minor.png'.format(out_dir_img), 
            bbox_inches='tight')


# In[31]:


from sklearn.manifold import MDS

embedding = MDS(n_components=2).fit_transform(arr_input_imp.T)

df_ref['mds_x'] = embedding[:,0]
df_ref['mds_y'] = embedding[:,1]

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_ref, x='mds_x', y='mds_y', hue='Tissue', palette="tab20", style='Assay')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)

plt.savefig('{}/mds_ref_tissue.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[32]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_ref, x='mds_x', y='mds_y', hue='MinorGroup', palette="tab20", style='Assay')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.savefig('{}/mds_ref_minor.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# ## prediction of references (depricated)

# ```MEnet predict -i MEnet_ref.csv --model /home/yyasumizu/nanoporemeth/MEnet/220417/model_params/1297.pickle --input_type table --output_dir predict_ref --plotoff```

# In[29]:


df_ref_pred = pd.read_csv('../MEnet/{}/predict_ref/cell_proportion_MinorGroup.csv'.format(version), index_col=0)


# In[30]:


# plt.figure(figsize=(30,10))
sns.clustermap(df_ref_pred, cmap='cividis', yticklabels=False,
            col_cluster=False, row_cluster=False, col_colors=[dict_colors[x] for x in  df_ref['Assay']], figsize=(30,10))
plt.savefig('{}/heatmap_ref_prediction.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[31]:


for g in df_ref['MinorGroup'].unique():
        sns.clustermap(df_ref_pred[df_ref[df_ref['MinorGroup'] == g].index], cmap='cividis', yticklabels=True,
                col_cluster=False, row_cluster=False, col_colors=[dict_colors[x] for x in  df_ref['Assay']], figsize=(10,10))
        plt.title(g)
        plt.savefig('{}/heatmap_ref_prediction_{}.pdf'.format(out_dir_img,g), 
                bbox_inches='tight')


# In[ ]:




