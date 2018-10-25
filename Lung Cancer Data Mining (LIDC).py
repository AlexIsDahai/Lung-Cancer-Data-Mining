
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


url_pd = 'https://wiki.cancerimagingarchive.net/download/attachments/3539039/tcia-diagnosis-data-2012-04-20.xls?version=1&modificationDate=1334930231098&api=v2'
url_nsl= 'http://www.via.cornell.edu/lidc/list3.2.csv'
url_ncp = 'https://wiki.cancerimagingarchive.net/download/attachments/3539039/lidc-idri%20nodule%20counts%20%286-23-2015%29.xlsx?version=1&modificationDate=1435085651880&api=v2'


# In[3]:


df_pd = pd.read_excel(url_pd)
headers = ['TCIA Patient ID', 'Diagnosis at the Patient Level', 'Diagnosis Method', 
           'Primary tumor site for metastatic disease','Nodule 1 Diagnosis at the Nodule Level ',
          'Nodule 1 Diagnosis Method at the Nodule Level','Nodule 2 Diagnosis at the Nodule Level ',
          'Nodule 2 Diagnosis Method at the Nodule Level','Nodule 3 Diagnosis at the Nodule Level ',
          'Nodule 3 Diagnosis Method at the Nodule Level','Nodule 4 Diagnosis at the Nodule Level ',
          'Nodule 4 Diagnosis Method at the Nodule Level','Nodule 5 Diagnosis at the Nodule Level ',
          'Nodule 5 Diagnosis Method at the Nodule Level']
df_pd.columns = headers
df_pd.shape


# In[6]:


df_pd


# In[5]:


df_pd.dtypes


# In[7]:


df_pd.describe(include = 'all')


# In[8]:


df_pd.info()


# In[20]:


df_ncp = pd.read_excel(url_ncp)
df_ncp.shape


# In[ ]:


df_ncp_pd = pd.merge(df_ncp, df_pd, how='left', left_on='TCIA Patent ID', right_on='TCIA Patient ID')


# In[22]:


df_ncp_pd.head(69)


# #### Two datasets has been combined, initializing the 3rd one

# In[23]:


df_nsl = pd.read_csv(url_nsl)


# In[24]:


df_nsl.head(20)


# In[32]:


df_nsl.head()


# In[ ]:


## this is valuable because it has volume


# ## useless columns: Unnamed 8, column 1(index)

# In[ ]:


#Visualizng the dataset now


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[ ]:


df_nsl['roi'].plot(kind='pie')


# #### To Combined the 3rd dataset

# In[26]:


df_ncp_pd['Case ID'] = df_ncp_pd['TCIA Patent ID'].str[-4:]
df_ncp_pd['Case ID']=df_ncp_pd['Case ID'].apply(str)
df_nsl['case'] = df_nsl['case'].apply(str) 


# In[35]:


df_ncp_pd_nsl = pd.merge( df_nsl,df_ncp_pd, how='left', left_on='case', right_on='Case ID')
df_ncp_pd_nsl


# In[28]:


df_nsl.head()

Question:
1. What data field is not relavent in this analysis
2. How can I share this notebook like a Google Doc.
3. What are the 3 different cell type


4. Which variable should we put in this (really useful)
5. Why can't I see all 37 columns
6. if you already do this, how can I be any help?
7. why not other algorithms
8. Timeline in the overall project?
9. my asssignment?
# Sum of the nodule size, or biggest nodule
# Matrix multiplication 有关
# Which variable am I using?? 参考jinglu的分析可以照抄，做文献综述
# 
Q(Oct.24)
1. Any data standardization/normalization needed?