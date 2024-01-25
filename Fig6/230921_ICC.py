#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pingouin as pg

seed = 100
np.random.seed(seed)
get_ipython().run_line_magic('matplotlib', 'inline')
# %config InlineBackend.figure_format = 'retina'
from scipy import stats
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family']='sans serif'
plt.rcParams['font.sans-serif']='Arial'

pd.set_option('display.max_columns', 80)


# In[2]:


dir_img = '../MEnet_pred/230228/1927/img/'


# In[3]:


df_cli = pd.read_excel('../data/2020 ICC sample_arranged.xlsx')


# In[4]:


df_cli[df_cli['Sample ID'].duplicated()]


# In[5]:


df_cli = df_cli.drop_duplicates(subset='Sample ID')


# In[6]:


df_res = pd.read_csv('../MEnet_pred/230228/1927/cell_proportion_MinorGroup_allsamples.csv', index_col=0).T
df_res = df_res.loc[df_res.index.str.startswith('hICC')]

df_res.head(10)


# In[7]:


df_res.loc[['hICC1_N', 'hICC2_N', 'hICC5_N']].T.plot.bar(figsize=(12,2))
plt.title('Non-Tumor')
sns.despine()
plt.savefig(dir_img+'bar_nontumor.pdf', bbox_inches='tight', transparent=True)


# In[8]:


df_res.loc[['hICC1_T', 'hICC2_T', 'hICC5_T']].T.plot.bar(figsize=(12,2))
plt.title('Tumor')
sns.despine()
plt.savefig(dir_img+'bar_tumor.pdf', bbox_inches='tight', transparent=True)


# In[9]:


df_res = df_res.iloc[6:]
df_res['Sample ID'] = df_res.index

list_celltype = list(df_res.columns)
list_celltype.remove('Sample ID')
df_cli['Sample ID'] = 'hICC_' + df_cli['Sample ID'].astype(str).str.zfill(2)
df = pd.merge(df_res, df_cli, on='Sample ID', how='inner')
df.index = df['Sample ID']

df.head()


# import re
# 
# th_p = 0.05
# th_log2_abs = 0.2
# 
# 
# df_target = pd.read_csv('../data/hWGBS_ICC/hICC_bam/cnvkit/hg38.target.bed', sep='\t', header=None)
# df_target['region'] = df_target[0] + '_' + df_target[3]
# df_target = df_target[~df_target[0].isin(['chrX', 'chrY'])]
# # df_target = df_target[['region']]
# df_target = df_target.drop_duplicates(subset='region')
# df_target = df_target[~df_target['region'].str.contains(',')]
# df_target = df_target.reset_index(drop=True)
# df_target['length'] = df_target[2] - df_target[1]
# df_target = df_target[[0,3,'region', 'length']]
# df_target.columns = ['chr', 'region', 'regionid', 'length']
# df_target.head()
# 
# for s in df_res.index:
#     print(s)
# 
#     pattern = re.compile(r'hICC_0(\d)')
#     s_c = pattern.sub(r'hICC_\1', str(s))
# 
#     d_call = pd.read_csv('../data/hWGBS_ICC/hICC_bam/cnvkit/{}.chr1-22XY.sorted.call.cns'.format(s_c), sep='\t')
#     d_call = d_call[(d_call['log2'].abs() > th_log2_abs) & (d_call['p_ttest'] < th_p)]
#     d_call = d_call[~d_call['chromosome'].isin(['chrX', 'chrY'])]
# 
#     df_target[s] = 0
#     for pos,row in d_call.iterrows():
#         for g in row['gene'].split(','):
#             df_target.loc[(df_target.chr==row['chromosome']) & (df_target.region==g), s] = row['log2']
# 
# df_target = df_target.set_index('regionid').T
# 
# l_length = []
# for s in [x for x in df_target.index if 'hICC' in x]:
#     l_length.append(df_target.loc['length', df_target.loc[s].abs() > 0].sum())
# 
# df_target = df_target.loc[[x for x in df_target.index if 'hICC' in x]]
# df_target['total chrom abnormality'] = l_length
# 
# # df_target.index = df_target['Sample ID']
# 
# # df_target = df_target.reset_index()
# # df_target.index = ['Sample ID'] + list(df_target.index[1:])
# # df_target.columns = ['Sample ID'] + list(df_target.columns[1:])
# 
# df = pd.merge(df, df_target, left_index=True, right_index=True, how='inner')
# # df.loc[:,df.columns.str.startswith('chr')] = df.loc[:,df.columns.str.startswith('chr')].astype(float)
# # df = df[~df.lifespan.isna()]

# In[10]:


list_d = []
for s in df.index:
    f = f"/home/yyasumizu/nanoporemeth/data/hWGBS_ICC/hICC_bam/ichorCNA/{s.replace('hICC_0', 'hICC_')}.params.txt"

    data_dict = {}
    with open(f, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if ":" in line:
                key_value = line.strip().split(":")
                key = key_value[0].strip()
                value = key_value[1].strip()
                try:
                    value = float(value)  # try to convert to float
                except ValueError:
                    pass  # if not possible, keep as string

                data_dict[key] = value if value != "NA" else None
            
    list_d.append(pd.DataFrame(data_dict, index=[s]))

df = pd.merge(df, pd.concat(list_d), left_index=True, right_index=True)


# In[11]:


data_dict


# In[18]:


pg.corr(x=df['Liver'], y=df['Tumor Fraction'])


# In[19]:


pg.corr(x=df['Duct'], y=df['Tumor Fraction'])


# In[20]:


list_nontumor = ['Liver',
 'Fibroblast',
 'Adipocytes',
 'Endothelial-cells',
 'Cardiovascular',
 'Muscle',
 'Meg-Ery',
 'Neutrophils',
 'Eosinophils',
 'Monocytes',
 'Macrophage',
 'Dendritic-cells',
 'naive B-cells',
 'mature B-cells',
 'naive Treg',
 'activated Treg',
 'naive Tconv',
 'memory Tconv',
 'naive CTL',
 'memory CTL',
 'exausted CTL',
 'NK-cells']
df['nonTumorCells'] = df[list_nontumor].sum(axis=1)


# In[21]:


pg.corr(x=df[list_nontumor].sum(axis=1), y=df['Tumor Fraction'])


# In[22]:


# plt.figure(figsize=(3,3))
sns.lmplot(data=df, x='nonTumorCells', y='Tumor Fraction',height=3, aspect=1)

plt.savefig('../img/230228/1927/hICC/scatter_ichorCNA_nonTumor.pdf' , bbox_inches='tight', transparent=True)


# In[23]:


list(df.columns)


# In[24]:


df['HBsAg_bool'] = (df['HBsAg'] == 'Positive') * 1
df['HCVAb_bool'] = (df['HCVAb'] == 'Positive') * 1


# In[25]:


sns.heatmap(df[['HBsAg', 'HCVAb']]=='Positive')


# In[26]:


list(df.columns)


# In[27]:


dict_stage = {'I' : 0, 'II' : 1, 'III' : 2, 'IV-A': 3, 'IV-B': 4, np.nan: np.nan}
df['Stage_num'] = [dict_stage[x] for x in df['Stage']]


# In[28]:


sns.clustermap(df[list_celltype+['Tumor Fraction', 'Ploidy', 'lifespan', 'TimeToRecurrence', 'Gender_x',
 'Ascites',
 'Hepatic encephalopathy',
 'Nutritional status',
 'ICG R15',
 'ICG k',
 'Liver damage',
 'Child stage',
 'pough score',
 'Child-pough score',
 'Child-pough score.1',
 'AST(GOT)',
 'ALT(GPT)',
 'PLT',
 'T-Bil',
 'D-Bil',
 'ALB',
 'AFP(pre_surgery)',
 'AFP L3',
 'PIVKA-2(pre_surgery)',
 'CEA',
 'CA19-9',
 'PT%',
 'PT',
 'Stage_num', 'nonTumorCells',
 'HBsAg_bool', 'HCVAb_bool',
 'major diameter', 'minor diameter']].corr(), vmax=1, vmin=-1, cmap='seismic', center=0, figsize=(16,16))


# In[29]:


df[list_celltype+['Tumor Fraction', 'Ploidy', 'lifespan', 'TimeToRecurrence', 'Gender_x',
 'ICG R15',
 'AST(GOT)',
 'ALT(GPT)',
 'PLT',
 'T-Bil',
 'ALB',
 'AFP(pre_surgery)',
 'PIVKA-2(pre_surgery)',
 'CEA',
 'CA19-9',
 'PT',
 'Stage_num', 'nonTumorCells',
 'HBsAg_bool', 'HCVAb_bool',
 'major diameter', 'minor diameter']].to_csv('../img/230228/1927/hICC/df_numerical.csv')


# In[30]:


df_corr = pg.pairwise_corr(df, method='spearman', alternative='greater', padjust='bonf').round(3)


# In[31]:


df_corr


# In[32]:


df_corr[df_corr['p-corr']<0.05].head(200)


# In[33]:


list_d = []
for s in df.index:
    f = f"/home/yyasumizu/nanoporemeth/data/hWGBS_ICC/hICC_bam/ichorCNA/{s.replace('hICC_0', 'hICC_')}.cna.seg"
    d = pd.read_csv(f, sep='\t')

    list_d.append(d)
df_cna = pd.concat(list_d, axis=1)

df_cna.index = df_cna['chr'].iloc[:,0] + ':' + df_cna['start'].iloc[:,0].astype(str) + '-' + df_cna['end'].iloc[:,0].astype(str)


# In[34]:


df_cna


# In[35]:


plt.figure(figsize=(14,14))
sns.heatmap(df_cna[[x for x in df_cna.columns if 'copy.number' in x]], center=2, cmap='seismic')


# In[36]:


import matplotlib.cm as cm


# In[37]:


sns.clustermap(df[list_celltype], figsize=(12,8), cmap='viridis', row_colors=[cm.Blues(i/df['TimeToRecurrence'].max())
                                                                              if not np.isnan(i) else [0,0,0]
                                                                              for i in df['TimeToRecurrence']])


# In[38]:


sns.clustermap(df.sort_values(by='TimeToRecurrence')[list_celltype], figsize=(12,8), cmap='viridis', row_cluster=False,
              row_colors=[[cm.Blues(i/df['TimeToRecurrence'].max())
                          if not np.isnan(i) else [1,1,0]
                          for i in df.sort_values(by='TimeToRecurrence')['TimeToRecurrence']],
                         [cm.Reds(i/df['lifespan'].max())
                          if not np.isnan(i) else [1,1,0]
                          for i in df.sort_values(by='TimeToRecurrence')['lifespan']]])
plt.savefig('../img/230228/1927/hICC/cluster_timetoreccurence_lifespan.pdf', bbox_inches='tight')


# In[39]:


sns.clustermap(df.sort_values(by='TimeToRecurrence')[list_celltype], figsize=(12,8), cmap='seismic', row_cluster=False,
              row_colors=[[cm.Blues(i/df['TimeToRecurrence'].max())
                          if not np.isnan(i) else [1,1,0]
                          for i in df.sort_values(by='TimeToRecurrence')['TimeToRecurrence']],
                         [cm.Reds(i/df['lifespan'].max())
                          if not np.isnan(i) else [1,1,0]
                          for i in df.sort_values(by='TimeToRecurrence')['lifespan']]],
              z_score=1)
plt.savefig('../img/230228/1927/hICC/cluster_timetoreccurence_z.pdf', bbox_inches='tight')


# In[40]:


df


# In[41]:


sns.clustermap(df.sort_values(by='TimeToRecurrence')[list_celltype], figsize=(12,8), cmap='seismic', row_cluster=True,
              row_colors=[[cm.Blues(i/df['TimeToRecurrence'].max())
                          if not np.isnan(i) else [1,1,0]
                          for i in df.sort_values(by='TimeToRecurrence')['TimeToRecurrence']],
                         [cm.Reds(i/df['lifespan'].max())
                          if not np.isnan(i) else [1,1,0]
                          for i in df.sort_values(by='TimeToRecurrence')['lifespan']]],
              z_score=1)


# In[42]:


sns.clustermap(df.sort_values(by='lifespan')[list_celltype], figsize=(12,8), cmap='seismic', row_cluster=False,
              row_colors=[cm.Blues(i/df['lifespan'].max())
                          if not np.isnan(i) else [1,1,0]
                          for i in df.sort_values(by='lifespan')['lifespan']],
              z_score=1)
plt.savefig('../img/230228/1927/hICC/cluster_lifespan_z.pdf', bbox_inches='tight')


# In[43]:


from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter


# In[44]:


from sklearn import preprocessing


# In[45]:


df['TimeToRecurrence_E'] =  ~df['TimeToRecurrence'].isna() * 1
df.loc[df['TimeToRecurrence'].isna(), 'TimeToRecurrence'] = df.loc[df['TimeToRecurrence'].isna(), 'lifespan']
df = df[~df['TimeToRecurrence'].isna()]

df['lifespan_E'] = (df['status'] == 'dead') * 1
# df_cph = df[list_celltype + ['lifespan', 'lifespan_E']]
# # df_cph[list_celltype] = preprocessing.scale(df_cph[list_celltype])


# In[46]:


df_cph = df[list_celltype + ['AgeAtSugery', 'TimeToRecurrence', 'TimeToRecurrence_E', 'lifespan', 'lifespan_E']]
scaler = preprocessing.MinMaxScaler()
df_cph[list_celltype] = scaler.fit_transform(df_cph[list_celltype])

from sklearn.decomposition import NMF

n_components = 10
model = NMF(n_components=n_components, init='random', random_state=seed)
W = model.fit_transform(df_cph[list_celltype])
H = model.components_

df_H = pd.DataFrame(H, index=['NMF_{}'.format(i) for i in range(n_components)], columns=list_celltype)

for i in range(n_components):
    df_cph['NMF_{}'.format(i)] = W[:,i]
    df['NMF_{}'.format(i)] = W[:,i]


# In[47]:


cg_cells = sns.clustermap(df_H, figsize=(16,4), cmap='cividis', row_cluster=False, vmax=0.9)
plt.savefig('../img/230228/1927/hICC/heatmap.NMF.cells.pdf', bbox_inches='tight')


# In[48]:


cg_pts = sns.clustermap(df[['NMF_{}'.format(i) for i in range(n_components)]], cmap='cividis', figsize=(3,8), col_cluster=False, vmax=0.9)
plt.savefig('../img/230228/1927/hICC/heatmap.NMF.pt.pdf', bbox_inches='tight')


# In[49]:


df[['NMF_{}'.format(i) for i in range(n_components)]].max()


# In[50]:


df[['NMF_{}'.format(i) for i in range(n_components)]].min()


# In[51]:


sns.clustermap(df.loc[
    [df.index[i] for i in cg_pts.dendrogram_row.reordered_ind], 
    [list_celltype[i] for i in cg_cells.dendrogram_col.reordered_ind]], 
    cmap='seismic', center=0, z_score=1, row_cluster=False, col_cluster=False, figsize=(12,8))
plt.savefig('../img/230228/1927/hICC/heatmap.zscore.pdf', bbox_inches='tight')


# In[52]:


from PyComplexHeatmap import *


# In[53]:


df_H.max().max()


# In[60]:


cg_cells.dendrogram_col.reordered_ind


# In[77]:


max_indices = df_H.idxmax(axis=0).sort_values().index


# In[76]:


max_indices.sort_values()


# In[73]:


df_H


# In[78]:


sns.heatmap(df_H[max_indices])


# In[80]:


col_ha_dict={}
max_indices_cell = df_H.idxmax(axis=0).sort_values().index

for i in range(n_components):
    if i == 0:
            col_ha_dict['NMF{}'.format(i)] =         anno_simple(df_H.loc['NMF_{}'.format(i), max_indices_cell],
                    cmap='cividis', vmin=0, vmax=1, legend=True, height=3)
    else:
        col_ha_dict['NMF{}'.format(i)] =         anno_simple(df_H.loc['NMF_{}'.format(i), max_indices_cell],
                    cmap='cividis', vmin=0, vmax=1, legend=False, height=3)
    
col_ha = HeatmapAnnotation(**col_ha_dict, hgap=0, wgap=0, label_side='left',
    plot=True)


# In[84]:


df_W = df[df_H.index]
max_indices_sample = df_W.idxmax(axis=1).sort_values().index


# In[88]:


row_ha_dict={}
df_W = df[df_H.index]
max_indices_sample = df_W.idxmax(axis=1).sort_values().index

for i in range(n_components):
    row_ha_dict['NMF{}'.format(i)] =         anno_simple(df.loc[max_indices_sample, 'NMF_{}'.format(i)],
                    cmap='cividis', vmin=0, vmax=1, legend=False, height=4)
    
row_ha = HeatmapAnnotation(**row_ha_dict, hgap=0, wgap=0,axis=0,label_side='bottom',
    plot=True)


# In[89]:


df['status_Reccurence'] = ['Yes' if x == 1 else 'No' for x in df['TimeToRecurrence_E']]


# In[90]:


row_ha_left = HeatmapAnnotation(
    lifetime = anno_simple(df.loc[max_indices_sample, 'lifespan'],
                    cmap='Greens',legend=True, height=4),
    status = anno_simple(df.loc[max_indices_sample, 'status'].fillna('alive'),
                    legend=True, height=4, colors={'dead': 'grey', 'alive': 'white'}),
    TimeToReccurence = anno_simple(df.loc[max_indices_sample, 'TimeToRecurrence'],
                    cmap='Purples',legend=True, height=4),
    status_reccurence = anno_simple(df.loc[max_indices_sample, 'status_Reccurence'],
                                    colors={'Yes': 'grey', 'No': 'white'},
                    legend=True, height=4),

    hgap=0, wgap=0,axis=0,label_side='bottom',
    plot=True)


# In[91]:


plt.figure(figsize=(16, 8))
col_ha_dict={}

for i in range(n_components):
    if i == 0:
            col_ha_dict['NMF{}'.format(i)] =         anno_simple(df_H.loc['NMF_{}'.format(i), max_indices_cell],
                    cmap='cividis', vmin=0, vmax=1, legend=True, height=3)
    else:
        col_ha_dict['NMF{}'.format(i)] =         anno_simple(df_H.loc['NMF_{}'.format(i), max_indices_cell],
                    cmap='cividis', vmin=0, vmax=1, legend=False, height=3)
    
col_ha = HeatmapAnnotation(**col_ha_dict, hgap=0, wgap=0, label_side='left',
    plot=False)

row_ha_left = HeatmapAnnotation(
    lifetime = anno_simple(df.loc[max_indices_sample, 'lifespan'],
                    cmap='Greens',legend=True, height=4),
    status = anno_simple(df.loc[max_indices_sample, 'status'].fillna('alive'),
                    legend=True, height=4, colors={'dead': 'grey', 'alive': 'white'}),
    TimeToReccurence = anno_simple(df.loc[max_indices_sample, 'TimeToRecurrence'],
                    cmap='Purples',legend=True, height=4),
    status_reccurence = anno_simple(df.loc[max_indices_sample, 'status_Reccurence'],
                                    colors={'Yes': 'grey', 'No': 'white'},
                    legend=True, height=4),

    hgap=0, wgap=0,axis=0,label_side='bottom',
    plot=False)

cm = ClusterMapPlotter(data=df.loc[
    max_indices_sample, 
    max_indices_cell], 
    top_annotation=col_ha,right_annotation=row_ha,
    left_annotation=row_ha_left,
                       col_cluster=False,row_cluster=False,
                       show_rownames=False,show_colnames=True,
                       z_score=1,
                       tree_kws={'row_cmap': 'Set1'},verbose=0,legend_gap=5, center=0, vmin=-2,vmax=2,
                       cmap='RdYlBu_r',xticklabels_kws={'labelrotation':-90,'labelcolor':'black'})
plt.savefig('../img/230228/1927/hICC/complexheatmap.v2.pdf', bbox_inches='tight')


# In[60]:


from lifelines import CoxPHFitter

# Using Cox Proportional Hazards model
cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.01)
cph.fit(df_cph[['NMF_{}'.format(i) for i in range(n_components)]+['AgeAtSugery', 'lifespan', 'lifespan_E']], 'lifespan', event_col='lifespan_E')
cph.print_summary()


# In[61]:


plt.figure(figsize=(4,4))
cph.plot()
sns.despine()
plt.savefig('../img/230228/1927/hICC/HR.forest.life.pdf', bbox_inches='tight')


# In[62]:


cph.plot_partial_effects_on_outcome(covariates='NMF_4', values=[0, 0.2, 0.4, 0.6], cmap='coolwarm', figsize=(4,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
sns.despine()
plt.savefig('../img/230228/1927/hICC/km.life.NMF4.pdf', bbox_inches='tight')


# In[63]:


cph.plot_partial_effects_on_outcome(covariates='NMF_6', values=[0, 0.2, 0.4, 0.6], cmap='coolwarm', figsize=(4,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
sns.despine()
plt.savefig('../img/230228/1927/hICC/km.life.NMF6.pdf', bbox_inches='tight')


# In[64]:


cph.plot_partial_effects_on_outcome(covariates='NMF_6', values=[0, 0.2, 0.4, 0.6], cmap='coolwarm', figsize=(4,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
sns.despine()
plt.savefig('../img/230228/1927/hICC/km.life.NMF6.pdf', bbox_inches='tight')


# In[65]:


# Using Cox Proportional Hazards model
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.01)
cph.fit(df_cph[['NMF_{}'.format(i) for i in range(n_components)]+['AgeAtSugery', 'TimeToRecurrence', 'TimeToRecurrence_E']], 
        'TimeToRecurrence', event_col='TimeToRecurrence_E')
cph.print_summary()

plt.figure(figsize=(4,4))
cph.plot()
sns.despine()
plt.savefig('../img/230228/1927/hICC/HR.forest.rec.pdf', bbox_inches='tight')


# In[66]:


cph.plot_partial_effects_on_outcome(covariates='NMF_4', values=[0, 0.2, 0.4, 0.6, 0.8], cmap='coolwarm')


# In[67]:


dict_stage = {'I' : 0, 'II' : 1, 'III' : 2, 'IV-A': 3, 'IV-B': 4}


# In[68]:


from lifelines.utils import concordance_index


# In[69]:


# bootstrap

list_res = []
list_cindex = []
# prediction
df_cph = df[['NMF_{}'.format(i) for i in range(n_components)] + ['AgeAtSugery', 'lifespan', 'lifespan_E']]

for seed in range(100):
    ind_test = df_cph.sample(10, random_state=seed).index

    # Using Cox Proportional Hazards model
    cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.1)
    cph.fit(df_cph[~df_cph.index.isin(ind_test)], 'lifespan', event_col='lifespan_E')
    # cph.print_summary()

    # plt.figure(figsize=(4,10))
    # cph.plot()

    # predict median remaining life
    cph.predict_median(df_cph[df_cph.index.isin(ind_test)])

    df_test = df_cph[df_cph.index.isin(ind_test)]
    df_test['prediction'] = list(cph.predict_median(df_cph[df_cph.index.isin(ind_test)]))

    # df_test.plot.scatter(x='TimeToRecurrence', y='prediction')
    # print(stats.spearmanr(df_test['lifespan'], df_test['prediction']))
    res = stats.spearmanr(df_test['lifespan'], df_test['prediction'])
    list_res.append([res[0], res[1]])

    list_cindex.append(concordance_index(df_test['lifespan'], df_test['prediction'], df_test['lifespan_E']))
df_res = pd.DataFrame(list_res, columns=['r', 'p'])

df_cindex = pd.DataFrame(list_cindex, columns=['MEnet'])

plt.hist(list_cindex)


# bootstrap

list_res = []
list_cindex = []
# prediction
df_cph = df[['Stage', 'AgeAtSugery', 'lifespan', 'lifespan_E']]
df_cph['Stage'] = [dict_stage[x] for x in df_cph['Stage']]

for seed in range(100):
    ind_test = df_cph.sample(10, random_state=seed).index

    # Using Cox Proportional Hazards model
    cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.1)
    cph.fit(df_cph[~df_cph.index.isin(ind_test)], 'lifespan', event_col='lifespan_E')
    # cph.print_summary()

    # plt.figure(figsize=(4,10))
    # cph.plot()

    # predict median remaining life
    cph.predict_median(df_cph[df_cph.index.isin(ind_test)])

    df_test = df_cph[df_cph.index.isin(ind_test)]
    df_test['prediction'] = list(cph.predict_median(df_cph[df_cph.index.isin(ind_test)]))

    # df_test.plot.scatter(x='TimeToRecurrence', y='prediction')
    # print(stats.spearmanr(df_test['lifespan'], df_test['prediction']))
    res = stats.spearmanr(df_test['lifespan'], df_test['prediction'])
    list_res.append([res[0], res[1]])

    list_cindex.append(concordance_index(df_test['lifespan'], df_test['prediction'], df_test['lifespan_E']))
df_res = pd.DataFrame(list_res, columns=['r', 'p'])

plt.hist(list_cindex)

df_cindex['Stage'] = list_cindex


# In[70]:


df_cindex_stack = df_cindex.stack().reset_index()
df_cindex_stack.columns = ['_', 'param', 'c-index']

plt.figure(figsize=(4,4))
sns.swarmplot(data=df_cindex_stack, y='c-index', x='param')
plt.hlines(0.5,-0.5,1.5, color='k', linestyles='--')
sns.despine()
plt.savefig('../img/230228/1927/hICC/cindex.lifespan.pdf', bbox_inches='tight')

stats.mannwhitneyu(df_cindex['MEnet'], df_cindex['Stage'])


# In[72]:


# bootstrap

list_res = []
list_cindex = []
# prediction
df_cph = df[['NMF_{}'.format(i) for i in range(n_components)] + ['AgeAtSugery', 'TimeToRecurrence', 'TimeToRecurrence_E']]

for seed in range(100):
    ind_test = df_cph.sample(10, random_state=seed).index

    # Using Cox Proportional Hazards model
    cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.1)
    cph.fit(df_cph[~df_cph.index.isin(ind_test)], 'TimeToRecurrence', event_col='TimeToRecurrence_E')
    # cph.print_summary()

    # plt.figure(figsize=(4,10))
    # cph.plot()

    # predict median remaining life
    cph.predict_median(df_cph[df_cph.index.isin(ind_test)])

    df_test = df_cph[df_cph.index.isin(ind_test)]
    df_test['prediction'] = list(cph.predict_median(df_cph[df_cph.index.isin(ind_test)]))

    # df_test.plot.scatter(x='TimeToRecurrence', y='prediction')
    # print(stats.spearmanr(df_test['lifespan'], df_test['prediction']))
    res = stats.spearmanr(df_test['TimeToRecurrence'], df_test['prediction'])
    list_res.append([res[0], res[1]])

    list_cindex.append(concordance_index(df_test['TimeToRecurrence'], df_test['prediction'], df_test['TimeToRecurrence_E']))
df_res = pd.DataFrame(list_res, columns=['r', 'p'])

df_cindex = pd.DataFrame(list_cindex, columns=['MEnet'])

plt.hist(list_cindex)


# bootstrap

list_res = []
list_cindex = []
# prediction
df_cph = df[['Stage', 'AgeAtSugery', 'TimeToRecurrence', 'TimeToRecurrence_E']]
df_cph['Stage'] = [dict_stage[x] for x in df_cph['Stage']]

for seed in range(100):
    ind_test = df_cph.sample(10, random_state=seed).index

    # Using Cox Proportional Hazards model
    cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.1)
    cph.fit(df_cph[~df_cph.index.isin(ind_test)], 'TimeToRecurrence', event_col='TimeToRecurrence_E')
    # cph.print_summary()

    # plt.figure(figsize=(4,10))
    # cph.plot()

    # predict median remaining life
    cph.predict_median(df_cph[df_cph.index.isin(ind_test)])

    df_test = df_cph[df_cph.index.isin(ind_test)]
    df_test['prediction'] = list(cph.predict_median(df_cph[df_cph.index.isin(ind_test)]))

    # df_test.plot.scatter(x='TimeToRecurrence', y='prediction')
    # print(stats.spearmanr(df_test['lifespan'], df_test['prediction']))
    res = stats.spearmanr(df_test['TimeToRecurrence'], df_test['prediction'])
    list_res.append([res[0], res[1]])

    list_cindex.append(concordance_index(df_test['TimeToRecurrence'], df_test['prediction'], df_test['TimeToRecurrence_E']))
df_res = pd.DataFrame(list_res, columns=['r', 'p'])

plt.hist(list_cindex)

df_cindex['Stage'] = list_cindex


# In[73]:


df_cindex_stack = df_cindex.stack().reset_index()
df_cindex_stack.columns = ['_', 'param', 'c-index']

plt.figure(figsize=(4,4))
sns.swarmplot(data=df_cindex_stack, y='c-index', x='param')
plt.hlines(0.5,-0.5,1.5, color='k', linestyles='--')
sns.despine()
plt.savefig('../img/230228/1927/hICC/cindex.timetorec.pdf', bbox_inches='tight')

stats.mannwhitneyu(df_cindex['MEnet'], df_cindex['Stage'])

