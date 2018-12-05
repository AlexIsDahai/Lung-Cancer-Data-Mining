
# coding: utf-8

# In[1]:


#!conda install -c conda-forge pydotplus -y
#!conda install -c conda-forge python-graphviz -y


# ### Get Data 

# In[3]:


## Get datasets from local machine into Jupyter Pandas dataframe, check shape
import pandas as pd
import numpy as np
df_pt_full = pd.read_csv('/Users/dahailiu/Desktop/DM for Cancer /NLST dataset/participant.data.d100517.csv')
df_lc = pd.read_csv('/Users/dahailiu/Desktop/DM for Cancer /NLST dataset/Lung Cancer/lung_cancer.data.d100517.csv')
df_sctabn_full = pd.read_csv('/Users/dahailiu/Desktop/DM for Cancer /NLST dataset/Spiral CT Abnormalities/sct_abnormalities.data.d100517.csv')
print(df_pt_full.shape)
print(df_lc.shape)
print(df_sctabn_full.shape)
pd.set_option('display.max_columns', None)


# ### Dataset #1: patient dataset, df_pt

# In[9]:


#### Trim dataset: Selecting only the people being CT scanned, and those has a record of Cancer vs. Non Cancer
## Select only CT scan
df_pt_ct = df_pt_full.loc[df_pt_full['rndgroup'] == 1]
## Select only people has a record of cancer vs non cancer
df_pt = df_pt_ct.loc[df_pt_ct['conflc'].isin([1,2])]
print('Shape of this datset:', df_pt.shape)
## This will trim the patient dataset df_pt into 6379 rows, from the original of 53452 rows
## summarize the cancer/Non-cancer count in pt dataset
conflc_counts = df_pt['conflc'].value_counts().to_frame()
conflc_counts.rename(columns={'conflc': 'Number of patient'}, inplace=True)
conflc_counts.index.name = 'conflc'
conflc_counts
# 1089 with cancer, and 5290 without cancer


# ### Dataset #3 -Sct_abnormality: df_sctabn and its trim

# In[11]:


## Select in abnormality only the rows that has sct_ab_desc in 51,52,53,62

df_abn_bigsmallnomany = df_sctabn_full.loc[df_sctabn_full['sct_ab_desc'].isin([51,52,53,62])]

## select columns in abnormality dataset that's only useful for the research
df_abn_useful = df_abn_bigsmallnomany[['dataset_version','pid','sct_ab_desc','sct_ab_num', 'sct_epi_loc',
                                       'sct_long_dia','sct_slice_num','study_yr']] 
df_abn_useful.shape


# In[13]:


## Calculation of largest nodule, total nodule size, number of nodule
df_abn_agg = df_abn_useful[['pid','sct_long_dia','study_yr']]
df_abn_nodulesum =    df_abn_agg.groupby(['pid'], as_index = False).sum()
df_abn_nodulemax =    df_abn_agg.groupby(['pid'], as_index = False).max()
df_abn_nodulecounts = df_abn_agg.groupby(['pid'], as_index = False).count()
## 81,356 record will become 19,116 after picking the sum/max/count with one pid only appear once
## Renaming the 3 data frames with each of their 'sct_long_dia' columns to indicate it's a sum, max, or a count
df_abn_nodulesum.rename(columns={'sct_long_dia': 'sct_long_dia_sum'}, inplace=True)
df_abn_nodulemax.rename(columns={'sct_long_dia': 'sct_long_dia_max'}, inplace=True)
df_abn_nodulecounts.rename(columns={'sct_long_dia': 'sct_long_dia_count'}, inplace=True)


# ### Joining #1 df_pt with #3 df_sct_abn

# In[14]:


df_pt_abn_sum = pd.merge(df_pt, df_abn_nodulesum, how='left', left_on='pid', right_on='pid')
df_pt_abn_sum_max = pd.merge(df_pt_abn_sum, df_abn_nodulemax, how='left', left_on='pid', right_on='pid')
df_pt_abn_sum_max_counts = pd.merge(df_pt_abn_sum_max, df_abn_nodulecounts, how='left', left_on='pid', right_on='pid')
df_pt_abn_sum_max_counts.head()
## From here, the pt dataset is combined with abnormal dataset, with nodules size info, that contains max/sum/counts
## The N match exactly what has been discussed, which is 6379 patients has a record of Cancer/No Cancer
## Surprisingly, and happlily, the N=6379 has not been decreased by joining the abn dataset with sct_ab_desc in (51, 52, 53, 62)


# ### Assembling Table 1:

# In[25]:


## Building table 1 for the first model(this we only use dataset pt, and CT, but not lc)
## Based on literature review of Jinglu's code:

df_pt_abn = df_pt_abn_sum_max_counts[['pid', 'age','gender', 'smokelive','race','pkyr','smokework',
                                      'famfather','fammother','anyscr_has_nodule','conflc', 'sct_long_dia_sum'
                                      ,'sct_long_dia_max', 'sct_long_dia_count','study_yr','diagcopd']]

## Did Jinglu use max, sum or count??? Which COPD is important?? 
print('The shape of the final dataset is: ',df_pt_abn.shape)

print('How many missing values in each columns?', '\n', df_pt_abn.isnull().sum())


# ### I just deleted rows contain missing values here, which I'm sure this isn't right

# In[26]:


df_pt_abn.shape
df_pt_abn = df_pt_abn.dropna()


# In[30]:


## Correcting Data Types for modeling purpose
df_pt_abn['gender'] = df_pt_abn['gender'].astype('object')
df_pt_abn['smokelive'] = df_pt_abn['smokelive'].astype('object')
df_pt_abn['race'] = df_pt_abn['race'].astype('object')
df_pt_abn['smokework'] = df_pt_abn['smokework'].astype('object')
df_pt_abn['famfather'] = df_pt_abn['famfather'].astype('object')
df_pt_abn['fammother'] = df_pt_abn['fammother'].astype('object')
df_pt_abn['anyscr_has_nodule'] = df_pt_abn['anyscr_has_nodule'].astype('object')
df_pt_abn['conflc'] = df_pt_abn['conflc'].astype('int')
df_pt_abn['study_yr'] = df_pt_abn['study_yr'].astype('object')
df_pt_abn['diagcopd'] = df_pt_abn['diagcopd'].astype('object')
df_pt_abn['pkyr'] = df_pt_abn['pkyr'].astype('float32')
df_pt_abn['sct_long_dia_sum'] = df_pt_abn['sct_long_dia_sum'].astype('float32')
df_pt_abn['sct_long_dia_max'] = df_pt_abn['sct_long_dia_max'].astype('float32')
df_pt_abn['sct_long_dia_count'] = df_pt_abn['sct_long_dia_count'].astype('float32')
df_pt_abn['pid'] = df_pt_abn['pid'].astype('int32')
df_pt_abn['age'] = df_pt_abn['age'].astype('int32')
df_pt_abn.dtypes


# ### Data Examination

# In[29]:


a = list(['gender', 'smokelive','race','smokework',
                                      'famfather','fammother','anyscr_has_nodule','conflc'
                                      ,'sct_long_dia_max', 'sct_long_dia_count','study_yr','diagcopd'])
for column in a:
    print(df_pt_abn[column].value_counts())
    print('                              ')


# In[28]:


export_path = '/Users/dahailiu/Downloads/20181127_1226.csv'
#df_pt_abn.to_csv(export_path)


# ## Machine Learning starts here:

# In[31]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# ### (Here is how Jinglu measured 3 times of feature importance, which will inform feature selection)

# In[32]:


#Attribute usage:
# 100.00% RaceCategory #100.00% Ethnicity
#100.00% PackYears
#99.99% Age
#99.51% FamilyHxFather #99.34% Sex
#99.25% FamilyHxMother #98.57% SmokingNowCategory


# 100.00% Ethnicity #100.00% PackYears
#99.71% Age
#97.78% RaceCategory 
#84.81% FamilyHxMother 
#80.36% SmokingNowCategory 
#74.82% SecondSmokeAtHome 
#72.75% FamilyHxFather #64.44% Sex

#MeanDecreaseGini
#Age 1335.7826
#Sex 201.7961
#PackYears 1963.1295
#AbnormalCTdiametersize 916.6168
#AbnormalCTnumberofsuspiciousmasses 703.6684
#AbnormalCTtype   0.0000
#RaceCategory 293.6279
#EthnicityCategory 104.7826
#FamilyHxFather 186.8981
#FamilyHxMother 169.8810
#SecondSmokeAtHome 219.8933
#SecondSmokeAtWork 167.0132
#SmokingNowCategory 204.0901


# In[33]:


## Train-Test Split
df_pt_abn = df_pt_abn.reset_index()
from sklearn.model_selection import train_test_split
X = df_pt_abn [[ 'age','gender', 'smokelive','race','pkyr','smokework',
                                      'famfather','fammother','anyscr_has_nodule', 'sct_long_dia_sum'
                                      ,'sct_long_dia_max', 'sct_long_dia_count','diagcopd']]
y = df_pt_abn ['conflc']
y = y.astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[35]:


## Fit a decision Tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
tree2 = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train)
tree2.score(X_test, y_test)


# In[36]:


## Plotting Decision Tree
lpy = [item for item in X_train.columns]
import pydot_ng as pydot
from IPython.display import IFrame
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
with open("dt.dot","w") as dot_data:
    export_graphviz(tree2, out_file=dot_data, filled=True, 
                feature_names = lpy,label = 'all')
pydot.graph_from_dot_file("dt.dot").write_png("dt.png")
IFrame("dt.png", width = 1000, height = 500)


# ### Work undone: Plotting Feature Importance??

# ### GBDT

# In[37]:


from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
GBDT = GradientBoostingClassifier(learning_rate = .1, max_depth = 4, random_state = 0)
gbdt = GBDT.fit(X_train, y_train)
gbdt.score(X_test, y_test)


# In[38]:


## Confusion Matrics for Gradient Boosted Decision Tree
from sklearn.metrics import confusion_matrix
gbdt_predicted = gbdt.predict(X_test)
confusion_gbdt = confusion_matrix(y_test, gbdt_predicted)
print('gradient boost decision tree classifier',  confusion_gbdt)


# In[39]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall) 
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, gbdt_predicted)))
print('Precision, which matters more: {:.2f}'.format(precision_score(y_test, gbdt_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, gbdt_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, gbdt_predicted)))


# ### Plotting ROC Curve

# In[ ]:


## Plotting ROC Curves 这个会报错啊啊啊啊啊啊啊！！！！！
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
y_score_gbdt = gbdt.decision_function(X_test)
fpr_gbdt, tpr_gbdt, _ = roc_curve(y_test, y_score_gbdt)
roc_auc_gbdt = auc(fpr_gbdt, tpr_gbdt)
plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_gbdt, tpr_gbdt, lw=3, label='GBDT ROC curve (area = {:0.2f})'.format(roc_auc_gbdt))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# # Comments
Topic for Next meeting:
- How to handle missing values
- Yes, there are 324 columns in the first file, True n: 1089 + 5290 = 6379 Data pre-processing(This number is acheived): Only give me :1) 6379个人有conflc==1 or 2，是CT scan，2）Add up to generate the sum, and also the largest, just like the LIDC
- I have figure out the ratio of cancer vs. Non-cancer
- Make sure, the sharing of code works
- ## Did Jinglu use max, sum or count??? Which COPD is important?? 
- 取出table 1, output， 检查一下dataset，missing啊，datatype啊之类的Things Undone:
- Push to Git
- 先Join起来，后两个dataset都有studyr Try to join 3rd dataset
- 然后高要求才是，把第二个lc dataset也join进去，我的方案（这个可以问）是不是把最后一年的取出来？？
- Continue Michighan and IBM Data Analysis Course
- 代码： C5.0, AUC, 
- Do I need to make pid the index?
- 把几个论文里标亮的地方重新看一遍
- Compare logistic Regression??Finished
- Learning: Pandas Course, K-Fold validation, Decision Tree, GBDT, RF, IBM course,  Do I need to make pid the index? (Dementionality reduction) 
- implement first decision tree and initiate k-fold validation    - Important to notice: 开始没有癌症，不等于eventually没有癌症.They scan for 1-2 yrs and followed up for 8 years, s所以要把studyr＊8 拽出来
- 机器学习特征： Age, Race, Packyr, NLST_COPD(?), AbnormalCTdiametersize, AbnormalCTnumberofsuspiciousmasses, (or sct_long_dia in 3rd dataset),Family history, Smoking now, smoke at work, sex, mother, at home, Beat 85% accuracy
课后作业：


- What's the expectation for my work by Sunday? Can we study together
    1. [Done]Left-join the 3rd dataset (Abnormal CT Findings)[Done]
    2. [Done]Figure out the true number of "confirmed cancers" for CT patients (1089, 2058, 2150, ???)[Done]: It is comfirmed to be 1089 Cancer vs 5290 No Cancer
    3. [Done]Calculate 4 new columns based on what you did with the LIDC dataset (small nodules, big nodules, total nodules, total size)
    Decision tree
    Random Forest
    C5.0
    AUC, evaluate performance, is it above 80%? 85%? What are the top features?(with Importance score)
# # Keep some distance! Here's about the 3rd dataset

# In[ ]:


df_lc.head()


# In[ ]:


#check duplicate pid in df_lc
lcid = df_lc["pid"]
df_lc[lcid.isin(lcid[lcid.duplicated()])].head(15)

Why is pid duplicated here? 1) T0-T7 study yrs, and 2) Multiple cancers
# In[ ]:


## Check how many pid actually exist in this data frame
df_lc['pid'].nunique()
## Exactly 2058 patient discussed in the last meeting, as compared to 2150 in this dataset


# In[ ]:


df_pt_pd = pd.merge(df_pt, df_lc, how='left', left_on='pid', right_on='pid')

df_pt_pd.columns
df_pt_pd.dtypes
df_pt_pd.shape
df_pt_pd.info


# In[ ]:


## Select Useful Columns
selected_features  = ['pid','age','cigar','pkyr','smokelive', 'can_scr', 'famfather','fammother']
#'lesionsize_y' is not included now
df = df_pt_pd[selected_features]
df.shape


# In[ ]:


## Check duplication of pid
ids = df["pid"]
df[ids.isin(ids[ids.duplicated()])].shape

