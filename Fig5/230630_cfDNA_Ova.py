#!/usr/bin/env python
# coding: utf-8

# In[17]:


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


# In[18]:


from scipy.stats import pearsonr


# In[19]:


import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# In[20]:


df = pd.read_excel('./GYN_combined-csv-files_Minor Sophia_v3.xlsx', index_col=0).T 


# In[21]:


df.index


# In[22]:


selected_columns = ["Tumor Fraction","Ovary","Colon"]
index_list = ['GYN415', 'GYN417', 'GYN425', 'GYN426', 'GYN457', 'GYN460', 'GYN462',
       'GYN463', 'GYN468', 'GYN447', 'GYN442', 'GYN435',
       'GYN429']
subset_df = df.loc[index_list]
selected_ova = subset_df[selected_columns]


# In[23]:


sns.lmplot(x='Ovary', y='Tumor Fraction', data=selected_ova, fit_reg=False)

pearson_coefficient, p_value = pearsonr(selected_ova['Tumor Fraction'], selected_ova['Ovary'])

plt.text(0.3, 0.9, f'R = {pearson_coefficient:.2f}, P-value = {p_value:.2g}', ha='center', va='center', transform=plt.gca().transAxes)

plt.show()


# In[ ]:




