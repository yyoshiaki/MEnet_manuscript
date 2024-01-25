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

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

pd.set_option('display.max_columns', 80)


# In[2]:


import subprocess
import glob
import os
import pandas as pd


# In[5]:


f_ref = '/home/yyasumizu/nanoporemeth/data/220507_ref_subsample.csv'
dir_subsample = '/home/yyasumizu/nanoporemeth/data/subsample'

seed = 0
list_props = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

version = '230228'
out_dir_img = '../MEnet/{}/vis'.format(version)

os.makedirs(out_dir_img, exist_ok=True)


# In[6]:


df_ref = pd.read_csv(f_ref)

df_ref.head()


# In[79]:


for pos,row in df_ref.iterrows():
    os.makedirs(dir_subsample + '/' + row['FileID'], exist_ok=True)


# In[80]:


row = df_ref.iloc[0]


# In[81]:


row


# In[82]:


for prop in list_props:
    prop


# In[9]:


prop = 0.01


# ## subsample

# In[10]:


if row['Layout'] == 'PE':
    f_in = dir_subsample+'/'+ row['FileID'] + '_1.fastq.gz'
    f_out = dir_subsample+'/'+ row['FileID'] + '/'+ row['FileID'] + '.p' + str(prop) + '.s' + str(seed) + '_1.fastq'

    if not os.path.exists(f_out+'.gz'):
        cmd = 'zcat {f_i} | seqkit sample -p {p} -s {s} > {f_o}'.format(p=prop, s=seed, f_i=f_in, f_o=f_out)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = 'pigz {}'.format(f_out)
        print(cmd)
        # subprocess.run(cmd, shell=True)

    f_in = dir_subsample+'/'+ row['FileID'] + '_2.fastq.gz'
    f_out = dir_subsample+'/'+ row['FileID'] + '/'+ row['FileID'] + '.p' + str(prop) + '.s' + str(seed) + '_2.fastq'

    if not os.path.exists(f_out+'.gz'):
        cmd = 'zcat {f_i} | seqkit sample -p {p} -s {s} > {f_o}'.format(p=prop, s=seed, f_i=f_in, f_o=f_out)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = 'pigz {}'.format(f_out)
        print(cmd)
        # subprocess.run(cmd, shell=True)

else:
    f_in = dir_subsample+'/'+ row['FileID'] + '.fastq.gz'
    f_out = dir_subsample+'/'+ row['FileID'] + '/'+ row['FileID'] + '.p' + str(prop) + '.s' + str(seed) + '.fastq'

    if not os.path.exists(f_out+'.gz'):
        cmd = 'zcat {f_i} | seqkit sample -p {p} -s {s} > {f_o}'.format(p=prop, s=seed, f_i=f_in, f_o=f_out)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = 'pigz {}'.format(f_out)
        print(cmd)
        # subprocess.run(cmd, shell=True)


# ## bismark

# In[12]:


if row['Layout'] == 'PE' and row['Assay'] == 'WGBS':
    prefix = dir_subsample+'/'+ row['FileID'] + '/'+ row['FileID'] + '.p' + str(prop) + '.s' + str(seed)

    if not os.path.exists('{p}.bis.cov.gz'.format(p=prefix)):
        cmd = "trim_galore --paired {p}_1.fastq.gz {p}_2.fastq.gz --cores 4".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark --genome ~/reference/bismark/Gencode_v34/fasta -1 {p}_1_val_1.fq.gz -2 {p}_2_val_2.fq.gz --multicore 20".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "deduplicate_bismark --bam {p}_1_val_1_bismark_bt2_pe.bam".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark_methylation_extractor --gzip --multicore 20 --comprehensive --bedGraph {p}_1_val_1_bismark_bt2_pe.deduplicated.bam".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "cp {p}_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz {p}.bis.cov.gz".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

elif row['Layout'] == 'SE' and row['Assay'] == 'WGBS':
    prefix = dir_subsample+'/'+ row['FileID'] + '/'+ row['FileID'] + '.p' + str(prop) + '.s' + str(seed)

    if not os.path.exists('{p}.bis.cov.gz'.format(p=prefix)):
        cmd = "trim_galore --paired {p}.fastq.gz --cores 4".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark --genome ~/reference/bismark/Gencode_v34/fasta {p}_trimmed.fq.gz --multicore 20".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "deduplicate_bismark --bam {p}_trimmed_bismark_bt2.bam".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark_methylation_extractor --gzip --multicore 20 --comprehensive --bedGraph {p}_trimmed_bismark_bt2.deduplicated.bam".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "cp {p}_trimmed_bismark_bt2.deduplicated.bismark.cov.gz {p}.bis.cov.gz".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

elif row['Layout'] == 'PE' and row['Assay'] == 'RRBS':
    prefix = dir_subsample+'/'+ row['FileID'] + '/'+ row['FileID'] + '.p' + str(prop) + '.s' + str(seed)

    if not os.path.exists('{p}.bis.cov.gz'.format(p=prefix)):
        cmd = "trim_galore --paired {p}_1.fastq.gz {p}_2.fastq.gz --cores 4 --rrbs".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark --genome ~/reference/bismark/Gencode_v34/fasta -1 {p}_1_val_1.fq.gz -2 {p}_2_val_2.fq.gz --multicore 20".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark_methylation_extractor --gzip --multicore 20 --comprehensive --bedGraph {p}_1_val_1_bismark_bt2_pe.bam".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "cp {p}_1_val_1_bismark_bt2_pe.bismark.cov.gz {p}.bis.cov.gz".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)
    
elif row['Layout'] == 'SE' and row['Assay'] == 'WGBS':
    prefix = dir_subsample+'/'+ row['FileID'] + '/'+ row['FileID'] + '.p' + str(prop) + '.s' + str(seed)

    if not os.path.exists('{p}.bis.cov.gz'.format(prefix)):
        cmd = "trim_galore --paired {p}.fastq.gz --cores 4 --rrbs".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark --genome ~/reference/bismark/Gencode_v34/fasta {p}_trimmed.fq.gz --multicore 20".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)

        cmd = "bismark_methylation_extractor --gzip --multicore 20 --comprehensive --bedGraph {p}_trimmed_bismark_bt2.bam".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True)
        
        cmd = "cp {p}_trimmed_bismark_bt2.bismark.cov.gz {p}.bis.cov.gz".format(p=prefix)
        print(cmd)
        # subprocess.run(cmd, shell=True) 


# In[13]:


row


# ## stats

# In[20]:


row = df_ref.iloc[0]


# In[21]:


if not os.path.exists('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID'])):
    cmd = 'seqkit stats {d}/{f}/*.fastq.gz -T > {d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID'])
    print(cmd)
    subprocess.run(cmd, shell=True)


# In[22]:


df_seq_stats = pd.read_csv('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID']), sep='\t')
df_seq_stats


# In[1]:


for pos, row in df_ref.iterrows():
    # if not os.path.exists('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID'])):
        cmd = 'seqkit stats {d}/{f}/*.fastq.gz -T > {d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID'])
        print(cmd)
        subprocess.run(cmd, shell=True)


# ## downstream

# In[41]:


import numpy as np
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

import subprocess
import glob
import os
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[42]:


f_ref = '/home/yyasumizu/nanoporemeth/data/230228_ref_subsample.csv'
f_cat = '/home/yyasumizu/nanoporemeth/data/230228_categories.csv'
dir_subsample = '/home/yyasumizu/nanoporemeth/data/subsample'

version = '230228'
num_study = '1927'
dir_out = '/home/yyasumizu/nanoporemeth/MEnet_pred/{}/{}'.format(version, num_study)
out_dir_img = '../MEnet/{}/vis'.format(version)

list_seed = [1,2,3,4]
# list_seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# list_props = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
list_props = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]


# In[43]:


df_ref = pd.read_csv(f_ref)

df_ref.head()


# In[44]:


# df_ref = df_ref[df_ref['Assay'] == 'RRBS']


# In[45]:


seed = 1
prop = 0.01
row = df_ref.iloc[1]


# In[46]:


df_seq_stats = pd.read_csv('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID']), sep='\t')
read_len = df_seq_stats.sum_len.sum()
df_seq_stats


# In[47]:


row = df_ref.loc[5]

f = '{}/subsample.bisseq/{}.p{}.s{}/cell_proportion_MinorGroup.csv'.format(dir_out, row['FileID'], prop, seed)

d = pd.read_csv(f, index_col=0)

plt.figure(figsize=(4,10))
sns.heatmap(d)


# In[48]:


l = []
for seed in list_seed:
    for prop in list_props:
        for pos,row in df_ref.iterrows():
            # print(pos, prop)
            df_seq_stats = pd.read_csv('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID']), sep='\t')
            df_seq_stats['ID'] = df_seq_stats.file.str.split('/').str.get(-1).str.split('_').str.get(0).str.split('.fastq').str.get(0)
            read_len = df_seq_stats.loc[df_seq_stats.ID == '{}.p{}.s{}'.format(row['FileID'], prop, seed), 'sum_len'].sum()

            f = '{}/subsample.bisseq/{}.p{}.s{}/cell_proportion_MinorGroup.csv'.format(dir_out, row['FileID'], prop, seed)

            d = pd.read_csv(f, index_col=0)
            v = d.loc[row['MinorGroup']][0]

            list_ans = [1 if x == row['MinorGroup'] else 0 for x in d.index]
            list_pred = list(d.iloc[:,0])

            rmse = np.sqrt(mean_squared_error(list_ans, list_pred))
            r2 = r2_score(list_ans, list_pred)

            l.append([row['FileID'], row['Assay'], read_len, seed, prop, v, rmse, r2])

df_acc = pd.DataFrame(l, columns=['FileID', 'Assay', 'length', 'seed', 'prop', 'Accuracy', 'RMSE', 'R2'])
df_acc.head()

df_acc_gb = df_acc.groupby(by=['FileID', 'prop']).mean()


# In[49]:


sns.scatterplot(data=df_acc, x="length", y="Accuracy", hue='Assay', style='FileID')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
for fi in df_ref.FileID:
    plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'Accuracy'], 
    c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_ref.loc[df_ref.FileID == fi, 'Assay'].iloc[0]])
plt.xscale('log')
plt.xlabel('length (bp)')


# In[50]:


sns.scatterplot(data=df_acc, x="length", y="R2", hue='Assay', style='FileID')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
for fi in df_ref.FileID:
    plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'R2'], 
    c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_ref.loc[df_ref.FileID == fi, 'Assay'].iloc[0]])
plt.xscale('log')
plt.xlabel('length (bp)')


# In[51]:


l = []
for _ in range(500):
    rand = np.random.random(39)
    rand = rand / rand.sum()
    l.append(np.sqrt(mean_squared_error(list_ans, rand)))
plt.hist(l, bins=50)


# In[52]:


sns.scatterplot(data=df_acc, x="length", y="RMSE", hue='Assay', style='FileID')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

d_th = (df_acc_gb.reset_index().groupby('FileID').min()[['RMSE']] * 1.5).reset_index()
d_th.columns = ['FileID', 'th_RMSE']
d = pd.merge(df_acc_gb.reset_index(), d_th, on='FileID', how='left')
d = d[d['th_RMSE'] > d['RMSE']].groupby(by='FileID').min()[['length']]

for fi in df_ref.FileID:
    plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'RMSE'], 
    c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_ref.loc[df_ref.FileID == fi, 'Assay'].iloc[0]])
    # plt.scatter(d.loc[fi,'length'], 0.2, marker='x',
    # c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_ref.loc[df_ref.FileID == fi, 'Assay'].iloc[0]])
plt.xscale('log')
plt.xlabel('length (bp)')
plt.hlines(np.mean(l), xmin=10**7, xmax=5*10**10, color='k', linestyle='--')
plt.title('Minor Group')
sns.despine()
plt.savefig('{}/subsample_Minorgroup.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[55]:


df_cat = pd.read_csv(f_cat)
df_cat


# In[56]:


l = []
for seed in list_seed:
    for prop in list_props:
        for pos,row in df_ref.iterrows():
            # print(pos, prop)
            df_seq_stats = pd.read_csv('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID']), sep='\t')
            df_seq_stats['ID'] = df_seq_stats.file.str.split('/').str.get(-1).str.split('_').str.get(0).str.split('.fastq').str.get(0)
            read_len = df_seq_stats.loc[df_seq_stats.ID == '{}.p{}.s{}'.format(row['FileID'], prop, seed), 'sum_len'].sum()

            f = '{}/subsample.bisseq/{}.p{}.s{}/cell_proportion_MinorGroup.csv'.format(dir_out, row['FileID'], prop, seed)

            d = pd.read_csv(f, index_col=0)
            d['Tissue'] = df_cat.set_index('MinorGroup').loc[d.index, 'Tissue']
            d = d.groupby(by='Tissue').sum()
            v = d.loc[row['Tissue']][0]

            list_ans = [1 if x == row['Tissue'] else 0 for x in d.index]
            list_pred = list(d.iloc[:,0])

            rmse = np.sqrt(mean_squared_error(list_ans, list_pred))
            r2 = r2_score(list_ans, list_pred)

            l.append([row['FileID'], row['Assay'], read_len, seed, prop, v, rmse, r2])

df_acc = pd.DataFrame(l, columns=['FileID', 'Assay', 'length', 'seed', 'prop', 'Accuracy', 'RMSE', 'R2'])
df_acc.head()

df_acc_gb = df_acc.groupby(by=['FileID', 'prop']).mean()


# In[57]:


sns.scatterplot(data=df_acc, x="length", y="Accuracy", hue='Assay', style='FileID')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
for fi in df_ref.FileID:
    plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'Accuracy'], 
    c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_ref.loc[df_ref.FileID == fi, 'Assay'].iloc[0]])
plt.xscale('log')
plt.xlabel('length (bp)')


# In[58]:


sns.scatterplot(data=df_acc, x="length", y="R2", hue='Assay', style='FileID')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
for fi in df_ref.FileID:
    plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'R2'], 
    c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_ref.loc[df_ref.FileID == fi, 'Assay'].iloc[0]])
plt.xscale('log')
plt.xlabel('length (bp)')


# In[59]:


l = []
for _ in range(500):
    rand = np.random.random(len(list_ans))
    rand = rand / rand.sum()
    l.append(np.sqrt(mean_squared_error(list_ans, rand)))
plt.hist(l, bins=50)


# In[60]:


sns.scatterplot(data=df_acc, x="length", y="RMSE", hue='Assay', style='FileID')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
for fi in df_ref.FileID:
    plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'RMSE'], 
    c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_ref.loc[df_ref.FileID == fi, 'Assay'].iloc[0]])
plt.xscale('log')
plt.xlabel('length (bp)')
plt.title('Major Group')
plt.hlines(np.mean(l), xmin=10**7, xmax=5*10**10, color='k', linestyle='--')
sns.despine()
plt.savefig('{}/subsample_majorgroup.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# In[61]:


np.sqrt(mean_squared_error(list_ans, list_pred))


# In[62]:


l = []
for _ in range(200):
    rand = np.random.random(len(list_pred))
    rand = rand / rand.sum()
    l.append(np.sqrt(mean_squared_error(list_ans, rand)))
plt.hist(l, bins=20)


# In[63]:


plt.hist(l, bins=20)


# ## Mix
# 
# extract several data points randomely and sum up using biscov 

# import random
# 
# n_sample = 3

# for pos,row in df_ref.iterrows():
#     # print(pos, prop)
#     d = pd.read_csv('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID']), sep='\t')
#     if pos == 0:
#         df_seq_stats = d
#     else:
#         df_seq_stats = pd.concat([df_seq_stats,d])
#         
# df_seq_stats['ID'] = df_seq_stats.file.str.split('/').str.get(-1).str.split('_').str.get(0).str.split('.fastq').str.get(0)

# df_seq_stats.head()

# d_sample = df_ref[df_ref['Assay'] == 'WGBS'].sample(n=n_sample)
# 
# list_len = []
# flag = 0
# for pos,row in d_sample.iterrows():
#     s = random.sample(list_seed, 1)[0]
#     p = random.sample(list_props, 1)[0]
#     read_len = df_seq_stats.loc[df_seq_stats.ID == '{}.p{}.s{}'.format(row['FileID'], p, s), 'sum_len'].sum()
#     list_len.append(read_len)
# 
#     d_bis = pd.read_csv('{}/{}/{}.p{}.s{}.bismark.cov.gz'.format(dir_subsample, row['FileID'], row['FileID'], p, s), header=None, sep='\t')
#     d_bis.columns = ['chr', 'start', 'end', '_', 'm_{}'.format(pos), 'u_{}'.format(pos)]
#     d_bis['posid'] = d_bis['chr'] + '_' + d_bis['end'].astype(str)
# 
#     if flag == 0:
#         flag = 1
#         df_bis = d_bis
#     else:
#         df_bis = pd.merge(df_bis, d_bis, on='posid', how='outer')
#         df_bis = df_bis[[x for x in df_bis.columns if x.startswith('m_') or x.startswith('u_')] + ['posid']]
# 
# df_bis['meth'] = df_bis[[x for x in df_bis.columns if x.startswith('m_')]].sum(axis=1)
# df_bis['unmeth'] = df_bis[[x for x in df_bis.columns if x.startswith('u_')]].sum(axis=1)
# df_bis['chr'] = df_bis.posid.str.split('_').str.get(0)
# df_bis['start'] = df_bis.posid.str.split('_').str.get(1)
# df_bis['rate'] = 100 * df_bis['meth'] / (df_bis['meth'] + df_bis['unmeth'])
# df_bis = df_bis[['chr', 'start', 'start', 'rate', 'meth', 'unmeth']]
# 
# d_sample['read_length'] = list_len

# In[93]:


dir_out = '/home/yyasumizu/nanoporemeth/MEnet_pred/{}/{}/subsample.bisseq.mix'.format(version, num_study)


# In[94]:


dir_out


# In[95]:


list_seed = [0,1,2,3,4]
list_props = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
list_assay = ['RRBS', 'WGBS']


list_ref_mix = []
for assay in list_assay:
    for p in list_props:
        for seed in list_seed:
            for pos,row in df_ref.iterrows():
                # print(pos, prop)
                d = pd.read_csv('{d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID']), sep='\t')
                if pos == 0:
                    df_seq_stats = d
                else:
                    df_seq_stats = pd.concat([df_seq_stats,d])
                    
            df_seq_stats['ID'] = df_seq_stats.file.str.split('/').str.get(-1).str.split('_').str.get(0).str.split('.fastq').str.get(0)

            # d_sample = df_ref[df_ref['Assay'] == assay].sample(n=n_sample, random_state=seed)
            d_sample = df_ref[df_ref['Assay'] == assay]
            # print(d_sample)

            list_len = []
            flag = 0
            # p = random.sample(list_props, 1)[0]

            for pos,row in d_sample.iterrows():
                # s = random.sample(list_seed, 1)[0]
                s = seed
                # p = random.sample(list_props, 1)[0]
                read_len = df_seq_stats.loc[df_seq_stats.ID == '{}.p{}.s{}'.format(row['FileID'], p, s), 'sum_len'].sum()

                assert read_len > 0, 'read_len was not found in {d}/{f}/seqstats.tsv'.format(d=dir_subsample, f=row['FileID'])

                list_len.append(read_len)
            d_sample['read_length'] = list_len

            prefix = assay
            for pos,row in d_sample.iterrows():
                prefix += "-" + row['FileID'] + ':' + str(row['read_length'])
            prefix = prefix[:-1]

            list_ref_mix.append([prefix, assay, p, seed])

df_ref_mix = pd.DataFrame(list_ref_mix, columns=['FileID', 'assay', 'prop', 'seed'])


# In[96]:


list_fin_files = glob.glob('{}/*/cell_proportion_MinorGroup.csv'.format(dir_out))
list_prefix = [x.split('/')[-2] for x in list_fin_files]


# In[97]:


l = []
l_true = []
l_pred = []
for prefix,f in tqdm(zip(list_prefix, list_fin_files)):
    d_sample = pd.read_csv(dir_subsample + '/mix2/' + prefix + '.table.csv', index_col=0)
    d = pd.read_csv(f, index_col=0)

    d.index = d.index.str.replace(' ', '')
    d_sample['MinorGroup'] = d_sample['MinorGroup'].str.replace(' ', '').str.replace('preGCB-cells', 'naiveB-cells')

    # list_ans = [1 if x == row['MinorGroup'] else 0 for x in d.index]
    arr_true = np.array([d_sample.set_index('MinorGroup').loc[x, 'read_length'] if x in list(d_sample.MinorGroup) else 0 for x in d.index])
    arr_true = arr_true / arr_true.sum()
    list_pred = list(d.iloc[:,0])

    l_true.append(list(arr_true))
    l_pred.append(list_pred)

    rmse = np.sqrt(mean_squared_error(arr_true, list_pred))
    r2 = r2_score(arr_true, list_pred)

    l.append([prefix, prefix.split('-')[0], d_sample['read_length'].sum(), seed, rmse, r2])

df_acc = pd.DataFrame(l, columns=['FileID', 'Assay', 'length', 'seed', 'RMSE', 'R2'])
df_acc["Nmix"] = [x.count(':') for x in df_acc.FileID]
df_acc = df_acc.sort_values(by=['Assay', 'length'])
df_acc = pd.merge(df_acc.reset_index(), df_ref_mix, on='FileID', how='left').set_index('index')
df_acc['seed'] = df_acc['seed_x']
print(df_acc.head())
df_acc_gb = df_acc.groupby(by=['assay', 'prop']).mean().reset_index()

df_true = pd.DataFrame(l_true, columns =d.index)
df_pred = pd.DataFrame(l_pred, columns =d.index)


# sns.scatterplot(data=df_acc, x="length", y="RMSE", hue='Assay', style='Nmix')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for pos,row in df_acc.iterrows():
#     plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[row['FileID'], 'RMSE'], 
#     c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_acc.loc[pos, 'Assay'].iloc[0]])
# plt.xscale('log')
# plt.xlabel('length (bp)')

# In[98]:


plt.figure(figsize=(10,4))
sns.heatmap(df_true.loc[df_acc[df_acc['Assay'] == 'WGBS'].index])


# In[99]:


plt.figure(figsize=(10,4))
sns.heatmap(df_pred.loc[df_acc[df_acc['Assay'] == 'WGBS'].index])


# In[100]:


plt.figure(figsize=(10,4))
sns.heatmap(df_true.loc[df_acc[df_acc['Assay'] == 'RRBS'].index])


# In[101]:


plt.figure(figsize=(10,4))
sns.heatmap(df_pred.loc[df_acc[df_acc['Assay'] == 'RRBS'].index])


# In[102]:


dict_l = {}
for assay in ['WGBS', 'RRBS']:
    l = []
    for _ in range(500):
        for pos,row in pd.DataFrame(l_true)[df_acc['Assay'] == assay].iterrows():
            rand = np.random.random(arr_true.shape[0])
            rand = rand / rand.sum()
            l.append(np.sqrt(mean_squared_error(list(row), rand)))
    dict_l[assay] = l
plt.hist(dict_l['RRBS'], bins=50, color='tab:blue')
plt.hist(dict_l['WGBS'], bins=50, color='tab:orange')


# In[103]:


df_acc_gb


# In[104]:


df_acc_gb


# In[107]:


sns.scatterplot(data=df_acc, x="length", y="RMSE", hue='Assay', hue_order=['WGBS', 'RRBS'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

for assay in ['RRBS', 'WGBS']:
    plt.plot(df_acc_gb.loc[df_acc_gb.assay==assay, 'length'], df_acc_gb.loc[df_acc_gb.assay==assay, 'RMSE'], 
    c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[assay])

plt.hlines(np.mean(dict_l['RRBS']), xmin=3*10**6, xmax=5*10**10, color='tab:orange', linestyle='--')
plt.hlines(np.mean(dict_l['WGBS']), xmin=3*10**6, xmax=5*10**10, color='tab:blue', linestyle='--')

plt.xscale('log')
plt.xlabel('length (bp)')
plt.title('Minor Group')

sns.despine()

plt.savefig('{}/subsample_mix_minorgroup.pdf'.format(out_dir_img), 
            bbox_inches='tight')


# sns.scatterplot(data=df_acc, x="length", y="R2", hue='Assay', style='Nmix')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for fi in df_acc.FileID:
#     plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'R2'], 
#     c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_acc.loc[df_acc.FileID == fi, 'Assay'].iloc[0]])
# plt.xscale('log')
# plt.xlabel('length (bp)')

# sns.scatterplot(data=df_acc, x="length", y="RMSE", hue='Assay', style='Nmix')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# for fi in df_acc.FileID:
#     plt.plot(df_acc_gb.loc[fi, 'length'], df_acc_gb.loc[fi, 'RMSE'], 
#     c={'WGBS' : "tab:blue", 'RRBS' : "tab:orange"}[df_acc.loc[df_acc.FileID == fi, 'Assay'].iloc[0]])
# plt.xscale('log')
# plt.xlabel('length (bp)')

# In[77]:


list_fin_files = glob.glob('{}/*/cell_proportion_MajorGroup.csv'.format(dir_out))
list_prefix = [x.split('/')[-2] for x in list_fin_files]

l = []
l_true = []
l_pred = []
for prefix,f in tqdm(zip(list_prefix, list_fin_files)):
    d_sample = pd.read_csv(dir_subsample + '/mix2/' + prefix + '.table.csv', index_col=0)
    d = pd.read_csv(f, index_col=0)

    # d['Tissue'] = df_cat.set_index('MinorGroup').loc[d.index, 'Tissue']
    # d = d.groupby(by='Tissue').sum()
    # v = d.loc[row['Tissue']][0]
    d_sample.Tissue = d_sample.Tissue.str.replace('Liver', 'Liver_Pancreas').str.replace('_Pancreas_Pancreas', '_Pancreas')

    d.index = d.index.str.replace(' ', '')
    # d.index = d.index.str.replace('Liver', 'Liver_Pancreas')
    d_sample['Tissue'] = d_sample['Tissue'].str.replace(' ', '')

    # list_ans = [1 if x == row['MinorGroup'] else 0 for x in d.index]
    arr_true = np.array([d_sample.set_index('Tissue').loc[x, 'read_length'] if x in list(d_sample.Tissue) else 0 for x in d.index])
    arr_true = arr_true / arr_true.sum()
    list_pred = list(d.iloc[:,0])

    l_true.append(list(arr_true))
    l_pred.append(list_pred)

    rmse = np.sqrt(mean_squared_error(arr_true, list_pred))
    r2 = r2_score(arr_true, list_pred)

    l.append([prefix, prefix.split('-')[0], d_sample['read_length'].sum(), seed, rmse, r2])

df_acc = pd.DataFrame(l, columns=['FileID', 'Assay', 'length', 'seed', 'RMSE', 'R2'])
df_acc["Nmix"] = [x.count(':') for x in df_acc.FileID]
df_acc = df_acc.sort_values(by=['Assay', 'length'])
df_acc = pd.merge(df_acc.reset_index(), df_ref_mix, on='FileID', how='left').set_index('index')
df_acc['seed'] = df_acc['seed_x']
print(df_acc.head())
df_acc_gb = df_acc.groupby(by=['assay', 'prop']).mean().reset_index()

# df_acc_gb = df_acc.groupby(by=['FileID', 'prop']).mean()


# In[78]:


df_true = pd.DataFrame(l_true, columns =d.index)
df_pred = pd.DataFrame(l_pred, columns =d.index)


# In[79]:


df_ref


# In[80]:


plt.figure(figsize=(8,4))
sns.heatmap(df_true.loc[df_acc[df_acc['Assay'] == 'RRBS'].index])


# In[81]:


plt.figure(figsize=(8,4))
sns.heatmap(df_pred.loc[df_acc[df_acc['Assay'] == 'RRBS'].index])


# In[82]:


plt.figure(figsize=(8,4))
sns.heatmap(df_true.loc[df_acc[df_acc['Assay'] == 'WGBS'].index])


# In[83]:


plt.figure(figsize=(8,4))
sns.heatmap(df_pred.loc[df_acc[df_acc['Assay'] == 'WGBS'].index])


# In[84]:


dict_l = {}
for assay in ['WGBS', 'RRBS']:
    l = []
    for _ in range(500):
        for pos,row in pd.DataFrame(l_true)[df_acc['Assay'] == assay].iterrows():
            rand = np.random.random(arr_true.shape[0])
            rand = rand / rand.sum()
            l.append(np.sqrt(mean_squared_error(list(row), rand)))
    dict_l[assay] = l
plt.hist(dict_l['RRBS'], bins=50, color='tab:blue')
plt.hist(dict_l['WGBS'], bins=50, color='tab:orange')


# In[92]:


sns.scatterplot(data=df_acc, x="length", y="RMSE", hue='Assay', hue_order=['WGBS','RRBS'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

for assay in ['RRBS', 'WGBS']:
    plt.plot(df_acc_gb.loc[df_acc_gb.assay==assay, 'length'], df_acc_gb.loc[df_acc_gb.assay==assay, 'RMSE'], 
    c={'RRBS' : "tab:orange", 'WGBS' : "tab:blue"}[assay])

    # # Calculate the first derivative of the losses
    # d_losses = np.diff(df_acc_gb.loc[df_acc_gb.assay==assay, 'RMSE'])
    # # Find the epoch where the rate of decrease is highest
    # elbow_length = list(df_acc_gb.loc[df_acc_gb.assay==assay, 'length'])[np.argmin(d_losses)+1]
    # print('elbow length: ', assay , elbow_length)

    # min_length = list(df_acc_gb.loc[df_acc_gb.assay==assay, 'length'])[np.argmin(df_acc_gb.loc[df_acc_gb.assay==assay, 'RSME'])]
    # print('min length: ', assay , min_length)

    # plt.scatter(min_length, 0.11)

plt.hlines(np.mean(dict_l['RRBS']), xmin=3*10**6, xmax=5*10**10, color='tab:orange', linestyle='--')
plt.hlines(np.mean(dict_l['WGBS']), xmin=3*10**6, xmax=5*10**10, color='tab:blue', linestyle='--')

plt.xscale('log')
plt.xlabel('length (bp)')
plt.title('Major Group')

sns.despine()

plt.savefig('{}/subsample_mix_majorgroup.pdf'.format(out_dir_img), 
            bbox_inches='tight')
