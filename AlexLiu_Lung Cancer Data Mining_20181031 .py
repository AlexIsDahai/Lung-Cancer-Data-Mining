
# coding: utf-8

# ### Import the first data set, clean the header, correct data type and alias

# In[1]:


# Make pandas display all columns
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


# In[2]:


# URL of the tree datasets
url_pd = 'https://wiki.cancerimagingarchive.net/download/attachments/3539039/tcia-diagnosis-data-2012-04-20.xls?version=1&modificationDate=1334930231098&api=v2'
url_nsl= 'http://www.via.cornell.edu/lidc/list3.2.csv'
url_ncp = 'https://wiki.cancerimagingarchive.net/download/attachments/3539039/lidc-idri%20nodule%20counts%20%286-23-2015%29.xlsx?version=1&modificationDate=1435085651880&api=v2'


# In[3]:


df_pd = pd.read_excel(url_pd)


# In[4]:


#Remove the notation of column headers to make it shorter
headers = ['TCIA Patient ID', 'Diagnosis at the Patient Level', 'Diagnosis Method', 
           'Primary tumor site for metastatic disease','Nodule 1 Diagnosis at the Nodule Level',
          'Nodule 1 Diagnosis Method at the Nodule Level','Nodule 2 Diagnosis at the Nodule Level',
          'Nodule 2 Diagnosis Method at the Nodule Level','Nodule 3 Diagnosis at the Nodule Level',
          'Nodule 3 Diagnosis Method at the Nodule Level','Nodule 4 Diagnosis at the Nodule Level',
          'Nodule 4 Diagnosis Method at the Nodule Level','Nodule 5 Diagnosis at the Nodule Level',
          'Nodule 5 Diagnosis Method at the Nodule Level']
df_pd.columns = headers


# In[5]:


# Making sure we are using the correct data type
df_pd['Diagnosis at the Patient Level'] = df_pd['Diagnosis at the Patient Level'].astype('object')
df_pd['Diagnosis Method'] = df_pd['Diagnosis Method'].astype('object')
df_pd['Nodule 1 Diagnosis at the Nodule Level'] = df_pd['Nodule 1 Diagnosis at the Nodule Level'].astype('object')


# In[6]:


# Make sure we are using the alias of the dataset(easly interpretable), not the coding of if
df_pd.loc[df_pd['Diagnosis at the Patient Level'] == 0, 'Diagnosis at the Patient Level'] = 'unknown'
df_pd.loc[df_pd['Diagnosis at the Patient Level'] == 1, 'Diagnosis at the Patient Level'] = 'benign or non-malignant disease'
df_pd.loc[df_pd['Diagnosis at the Patient Level'] == 2, 'Diagnosis at the Patient Level'] = 'malignant, primary lung cancer'
df_pd.loc[df_pd['Diagnosis at the Patient Level'] == 3, 'Diagnosis at the Patient Level'] = 'malignant metastatic'

df_pd.loc[df_pd['Diagnosis Method'] == 0, 'Diagnosis Method'] = 'unknown'
df_pd.loc[df_pd['Diagnosis Method'] == 1, 'Diagnosis Method'] = 'review of radiological images to show 2 years of stable nodule'
df_pd.loc[df_pd['Diagnosis Method'] == 2, 'Diagnosis Method'] = 'biopsy'
df_pd.loc[df_pd['Diagnosis Method'] == 3, 'Diagnosis Method'] = 'surgical resection'
df_pd.loc[df_pd['Diagnosis Method'] == 4, 'Diagnosis Method'] = 'progression or response'


df_pd.loc[df_pd['Nodule 1 Diagnosis at the Nodule Level'] == 0, 'Nodule 1 Diagnosis at the Nodule Level'] = 'unknown'
df_pd.loc[df_pd['Nodule 1 Diagnosis at the Nodule Level'] == 1, 'Nodule 1 Diagnosis at the Nodule Level'] = 'benign or non-malignant disease'
df_pd.loc[df_pd['Nodule 1 Diagnosis at the Nodule Level'] == 2, 'Nodule 1 Diagnosis at the Nodule Level'] = 'malignant, primary lung cancer'
df_pd.loc[df_pd['Nodule 1 Diagnosis at the Nodule Level'] == 3, 'Nodule 1 Diagnosis at the Nodule Level'] = 'malignant metastatic'

## 157 patient has diagnosis out of 1018


# ### Create Response Variable

# In[7]:


#Need to import numpy nan to fill in nan with "unknown" in patient diagnosis
from numpy import nan
df_pd.loc[df_pd['Diagnosis at the Patient Level']=='unknown', 'Cancer'] = nan
df_pd.loc[df_pd['Diagnosis at the Patient Level']=='benign or non-malignant disease', 'Cancer'] = 0
df_pd.loc[df_pd['Diagnosis at the Patient Level']=='malignant, primary lung cancer','Cancer' ] = 1
df_pd.loc[df_pd['Diagnosis at the Patient Level']=='malignant metastatic', 'Cancer'] = 1
df_pd['Cancer'] = df_pd['Cancer'].astype('object')
df_pd.shape


# ### Connecting 2nd dataset, merge, and keep records that has scanning result

# In[8]:


# Connecting 2nd dataset "nodule count by patient"
df_ncp_origin = pd.read_excel(url_ncp)
# Drop Unuseful Columns
df_ncp_origin = df_ncp_origin.drop(columns=['Unnamed: 4', 'Unnamed: 5'])
df_ncp_origin.shape
## Drop last row----it is a sum, not a patient
df_ncp = df_ncp_origin.dropna(subset = ['TCIA Patent ID'])
# Shape is (1018,4)


# In[9]:


## Merge df_ncp (Nodule count per patient) with df_pd (Patient Diagnosis), shape is (1018,19) (15+4 columns combined)
df_ncp_pd = pd.merge(df_ncp, df_pd, how='left', left_on='TCIA Patent ID', right_on='TCIA Patient ID')
df_ncp_pd.shape


# In[10]:


#keep records that only have a scanning result
df_ncp_pd = df_ncp_pd.dropna(subset = ['Cancer'])
df_ncp_pd.shape
# Generating dataframe with shape (131, 19)


# ### Connectin 3rd dataset df_nsl (nodule size list), clean and merge

# In[11]:


# connecting data source
df_nsl_full = pd.read_csv(url_nsl)
df_nsl_full.shape
# Selet columns that will only be used in the dataset
df_nsl = df_nsl_full[['case', 'scan','roi', 'volume']]
df_nsl.shape


# In[12]:


# Create column 'case' to be the trim of Patent ID with the same "four digit" format, and make both string type
df_ncp_pd['case'] = df_ncp_pd['TCIA Patent ID'].str[-4:]
df_ncp_pd['case'] = df_ncp_pd['case'].apply(str)
df_nsl['case'] = df_nsl['case'].apply(str) 


# In[13]:


# Merge df_nsl(nodule size list) with df_ncp_pd 
df_ncp_pd_nsl = pd.merge( df_nsl,df_ncp_pd, how='right', on='case')
df_ncp_pd_nsl


# Shape is (140, 19+4=23)


# In[14]:


# Select rows that has no nodule size or rows has largest nodule size
largest_volume = (df_ncp_pd_nsl['volume'] == df_ncp_pd_nsl.groupby(['case'])['volume'].transform(max)) 
no_nodule_size = df_ncp_pd_nsl['volume'].isnull()
df_ncp_pd_nsl.loc[ largest_volume | no_nodule_size]


# ## Dataset is finally prepared for modeling (we temporarily use ncp_pd 158*22 )

# In[15]:


df = df_ncp_pd_nsl.loc[ largest_volume | no_nodule_size]
#imput the missing nodule size
df['volume'] = df['volume'].fillna(0)
df


# ## Train-Test Split

# In[16]:


from sklearn.model_selection import train_test_split
import numpy as np
X = df [['Number of Nodules >=3mm**','Number of Nodules >=3mm**','Number of Nodules <3mm***']]
y = df ['Cancer']
## y can not be object
y=y.astype('int')

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ## Initiating Decision Tree

# In[17]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree

tree2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
print(tree2)


# In[18]:


tree2.score(X_test, y_test)


# ## Visualizing Decision Tree

# In[19]:


graph = Source(tree.export_graphviz(tree2, out_file=None
   , feature_names=labels, class_names=['0', '1', '2'] 
   , filled = True))

display(SVG(graph.pipe(format='svg')))


# Sum of the nodule size, or biggest nodule
# Matrix multiplication 有关
# Which variable am I using?? 参考jinglu的分析可以照抄，做文献综述
# 

# Question to be discussed:
# 
# 
# - 1) How to share code with Github and the tool ------------
# - 2) How to slice data frame by column--------------
# - 3) How to aggregate the data--------------
# - 4) What variables are used for the model----------
# - 5) How to put categorical data into decision tree-------------
# - 6) How to visualize decision tree
# - 7) The learning and testing will be 157 but later on 1018?
# - 8) How to put hyperlink to data
# - 9) precision and reproducability and hurt human?
# 

# - I imputed my nodule size
