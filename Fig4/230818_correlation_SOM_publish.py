#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy as np
from numpy.random import randn
import pandas as pd
import math
import pingouin as pg
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from IPython.display import display

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

pd.set_option('display.max_columns', 80)


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_excel("230810_HE_correlate_1927_SOM_retry.xlsx", sheet_name = "correlation_Lym (2)"
                     ,skiprows=[4,5,6,7,8,9,10,11,12,14,15,16,17] 
                     ,header=None,index_col=0).T
data


# In[3]:


data_ob = data[["Sample","FFPE_ID","がん種"]]

data_f = data[["MEnet","Lym"]]
data_f = data_f.astype(float)

df = data_ob.merge(data_f,left_index=True,right_index=True)
df


# In[ ]:





# In[14]:


from scipy.stats import pearsonr

g = sns.lmplot(x='MEnet', y='Lym', data=df_M, sharey=False, sharex=False,
               scatter_kws={'s': 100}, 
               line_kws={'lw': 4}) 

pearson_coefficient, p_value = pearsonr(df_M['MEnet'], df_M['Lym'])

plt.text(0.3, 0.9, f'R = {pearson_coefficient:.2f}, P-value = {p_value:.2g}', ha='center', va='center', transform=plt.gca().transAxes)

g.savefig(f'./230924_plot.pdf')  

plt.show()


# In[ ]:




