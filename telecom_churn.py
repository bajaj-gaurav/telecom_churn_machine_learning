#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# # Reading the data from the dataset

# In[3]:


telecom_churn = pd.read_csv("telecom_churn_data.csv")


# In[ ]:





# In[4]:


telecom_churn.describe()


# In[5]:


telecom_churn.columns


# # Adding churn label

# In[11]:


def churn_or_not(total_incoming, total_outgoing, total_2g, total_3g):
    
    if total_incoming == 0 and total_outgoing == 0 and total_2g == 0 and total_3g == 0: 
        return 1
    else: 
        return 0


telecom_high_value_cust['churn'] = telecom_high_value_cust.apply(lambda x: churn_or_not(x.total_ic_mou_9, x.total_og_mou_9, x.vol_2g_mb_9, x.vol_3g_mb_9), axis=1)




# # Structure of our dataset

# In[15]:


telecom_high_value_cust_without_9.head()


# # Removing the columns with 1 and all unique values except mobile number. As mobile number key identifier to find a customer. We can use this attribute to find the customers.

# In[16]:


col_1_unique_val = []
col_all_unique_val = []
col_without_single_or_all_unique = []
for i in telecom_high_value_cust_without_9.columns:
    temp_df = telecom_high_value_cust_without_9[i]
    temp_df = temp_df.dropna()
    if temp_df.unique().size == 1:
        col_1_unique_val.append(i)
    elif temp_df.unique().size == temp_df.size and i != "mobile_number":
        col_all_unique_val.append(i)
    else:
        col_without_single_or_all_unique.append(i)


# In[17]:


print(col_1_unique_val)
print(col_all_unique_val)
print(col_without_single_or_all_unique)


# In[18]:


telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9[col_without_single_or_all_unique]


# In[19]:


telecom_high_value_cust_without_9.shape


# # Dropping the rows with all the NaN values

# In[20]:


telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9.dropna(how='all')


# In[21]:


telecom_high_value_cust_without_9.shape


# # Removing columns/attributes corresponding to the amount as we want to know churning on the basis of the usage

# In[22]:


col_without_amount = [c for c in telecom_high_value_cust_without_9.columns if c[-5:-2] != "amt"]


# In[ ]:





# # Removing the avg recharge field which was calculated earlier

# In[23]:


col_without_amount.remove('avg_rech_amt_6_7')


# In[24]:


print(col_without_amount)


# In[25]:


telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9[col_without_amount]


# In[26]:


telecom_high_value_cust_without_9.shape


# # Checking for Missing Values and Inputing Them

# In[27]:


round(100*(telecom_high_value_cust_without_9.isnull().sum()/len(telecom_high_value_cust_without_9.index)), 2)


# # Filling all the missing values with 0 
# ## After studying the data, it was found that the columns which have null values are based on usage and internet packs, volume of usage and calls.
# ## We know that if there is no action taken for the above features then the column would have ideally the values -> 0. So, the assumption behind putting the values as 0 is customer has not done any action for those attributes.

# In[28]:


telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9.fillna(0)


# In[29]:


round(100*(telecom_high_value_cust_without_9.isnull().sum()/len(telecom_high_value_cust_without_9.index)), 2)


# In[ ]:





# In[30]:


telecom_high_value_cust_without_9.shape


# # Removing the date field as the customers who has not used the service for the fourth month has already churned

# In[31]:


col_without_date = [c for c in telecom_high_value_cust_without_9.columns if "date" not in c]


# In[32]:


print(col_without_date)


# In[33]:


telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9[col_without_date]
telecom_high_value_cust_without_9.shape


# In[34]:


telecom_high_value_cust_without_9.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# In[ ]:





# # Dropping high correlated attributes

# In[35]:


# if flag == all then we want the correlation for the similar month
def show_heatmap(cols, flag=None):
    #sns.pairplot(telecom_high_value_cust_without_9[cols], height = 5)
    plt.figure(figsize = (20,12))
    sns.heatmap(telecom_high_value_cust_without_9[cols].corr(), annot=True)
    if flag:
        month = ['_7', '_8']
        for mon in month:
            col_new = []
            for ele in cols:
                col_new.append(ele.replace("_6", mon))
            plt.figure(figsize = (20,12))
            sns.heatmap(telecom_high_value_cust_without_9[col_new].corr(), annot=True)
            
    
    


# In[36]:


cols = [<list of field names>
       ]
show_heatmap(cols)


# In[ ]:





# In[ ]:





# ## Dropping highly correlated fields

# In[40]:


telecom_high_value_cust_without_9.shape


# In[ ]:


## Dropping code here


# In[42]:


telecom_high_value_cust_without_9.shape


# In[43]:


telecom_high_value_cust_without_9.columns


# In[ ]:





# In[44]:


def show_heatmap_across_month(cols):
    #sns.pairplot(telecom_high_value_cust_without_9[cols], height = 5)
    month = ['_7', '_8']
    for ele in cols:
        col_new = []
        col_new.append(ele)
        for mon in month:
            col_new.append(ele.replace("_6", mon))
        plt.figure(figsize = (20,3))
        
        
        sns.heatmap(telecom_high_value_cust_without_9[col_new].corr(), annot=True)
        col_new.append("churn")
        print(telecom_high_value_cust_without_9[col_new].head(25))
            


# In[45]:


cols = ['xxx']
show_heatmap_across_month(cols)


# In[46]:


cols = ['xxx']
show_heatmap_across_month(cols)


# In[47]:


cols = ['xxx']
show_heatmap_across_month(cols)


# In[48]:


cols = ['xxx']
show_heatmap_across_month(cols)


# In[49]:


cols = ['xxx']
show_heatmap_across_month(cols)


# In[50]:


cols = ['xxx']
show_heatmap_across_month(cols)


# In[51]:


cols = ['xxx']
show_heatmap_across_month(cols)


# In[ ]:





# # Dividing the data into 2 phase: action and good. 

# ## The data of 6th and 7th month separately doesn't make any sense found after studying the data. As both are part of good phase, combining both of them and using their average

# In[ ]:





# In[52]:


telecom_high_value_cust_without_9.columns


# In[53]:


col_with_6 = [c for c in telecom_high_value_cust_without_9.columns if "_6" in c]


# In[54]:


col_with_7 = [c for c in telecom_high_value_cust_without_9.columns if "_7" in c]


# In[55]:


len(col_with_6)


# In[56]:


len(col_with_7)


# In[57]:


for element in col_with_6:
    target = element.replace("_6", "_6_7")
    var2 = element.replace("_6", "_7")
    telecom_high_value_cust_without_9[target] = (telecom_high_value_cust_without_9[element] + telecom_high_value_cust_without_9[var2])/2
    telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9.drop(element, axis=1)
    telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9.drop(var2, axis=1)


# In[58]:


telecom_high_value_cust_without_9.shape


# In[59]:


telecom_high_value_cust_without_9.columns


# In[60]:


telecom_high_value_cust_without_9['jun_jul_vbc_3g'] = (telecom_high_value_cust_without_9['jul_vbc_3g'] + telecom_high_value_cust_without_9['jun_vbc_3g'])/2
telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9.drop('jul_vbc_3g', axis=1)
telecom_high_value_cust_without_9 = telecom_high_value_cust_without_9.drop('jun_vbc_3g', axis=1)


# In[61]:


telecom_high_value_cust_without_9.shape


# In[62]:


telecom_high_value_cust_without_9.head()


# In[ ]:





# # Feature Standardisation except mobile number

# In[ ]:





# In[63]:


col_binary = []
col_non_binary = []
col_cust_id = ['mobile_number']
for i in telecom_high_value_cust_without_9.columns:
    temp_df = telecom_high_value_cust_without_9[i]
    temp_list = temp_df.unique()
    if temp_list.size == 2 and 1 in temp_list and 0 in temp_list:
        col_binary.append(i)
    else:
        if i != "mobile_number":
            col_non_binary.append(i)


# In[64]:


print(col_binary)
print(col_non_binary)
print(col_cust_id)


# In[65]:


df_to_normalise = telecom_high_value_cust_without_9[col_non_binary]
df_binary = telecom_high_value_cust_without_9[col_binary]
df_cust_id = telecom_high_value_cust_without_9[col_cust_id]


# In[66]:


normalized_df = (df_to_normalise-df_to_normalise.mean())/df_to_normalise.std()


# In[67]:


telecom = pd.concat([df_binary, normalized_df],axis=1)
telecom = pd.concat([df_cust_id, telecom], axis=1)
telecom.head()


# In[ ]:





# # The data cleaning and preprocessing is completed.

# In[ ]:





# # Checking the Churn Rate

# In[68]:


churn = (sum(telecom['churn'])/len(telecom['churn'].index))*100
churn


# ## We have low rate of churn. So, we will using a technique to handle the imbalance later

# # Model Building

# ## Splitting Data into Training and Test Sets

# In[69]:


#from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = telecom.drop(['churn', 'mobile_number'],axis=1)
#X = X_cust.drop(['mobile_number'],axis=1)

# Putting response variable to y
y_cust = telecom[['churn', 'mobile_number']]

y_cust.head()


# In[ ]:





# In[70]:


# Splitting the data into train and test
X_train, X_test, y_train_cust, y_test_cust = train_test_split(X,y_cust, train_size=0.7,test_size=0.3,random_state=100)


# In[71]:


X_train.head()


# In[ ]:





# # Dropping the mobile number from the Y

# In[72]:


y_train = y_train_cust.drop(['mobile_number'], axis=1)
y_test = y_test_cust.drop(['mobile_number'], axis=1)


# ### Over-sampling is to duplicate random records from the minority class, which can cause overfitting. In under-sampling, the simplest technique involves removing random records from the majority class, which can cause loss of information
# 
# 

# ## Out of the 4 explored options:
# #### Undersampling, 
# #### Oversampling, 
# #### Synthetic Data Generation, 
# #### Cost Sensitive Learning, 
# ### Synthetic Data Generation looks more suitable as it will be less prone to overfitting and also there will be no loss of data

# In[73]:


imbalance_train = churn = (sum(y_train['churn'])/len(y_train['churn'].index))*100
print("Telecom train dataset Imbalance before smote: {}".format(imbalance_train))


# In[74]:


# sampling_strategy: auto which is equivalent to "not majority" ie, oversampling all the classes except the majority
# kind: regular
smote = SMOTE(kind = "regular")
X_train_balanced,y_train_balanced = smote.fit_sample(X_train,y_train)


churn_percentage = (sum(y_train_balanced)/len(y_train_balanced))*100

print("X train dataset {}".format(X_train_balanced.shape))
print("y train dataset {}".format(y_train_balanced.shape))

print("Telecom train dataset Imbalance after smote: {}".format(churn_percentage))


# In[75]:


print(type(X_train_balanced))
print(type(X_train))
print(type(y_train_balanced))
print(type(y_train))


# In[76]:


X_train_balanced = pd.DataFrame(X_train_balanced)
y_train_balanced = pd.DataFrame(y_train_balanced)
print(type(y_train_balanced))
print(type(X_train_balanced))


# In[77]:


X_train.head()


# # Converting the numeric headers into the original ones

# In[78]:


X_train_balanced.columns = X_train.columns
X_train_balanced.head()


# In[79]:


y_train_balanced.head()


# In[80]:


y_train.head()


# In[81]:


y_train_balanced.columns = y_train.columns
y_train_balanced.head()


# # Running logistics regression

# In[82]:


import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics


# In[83]:


logreg = LogisticRegression()

rfe = RFE(logreg, 8)             # running RFE with 8 variables as output
rfe = rfe.fit(X_train_balanced,y_train_balanced)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)  


# In[84]:


print(rfe.support_)


# In[85]:


print(list(zip(X_train_balanced.columns, rfe.support_, rfe.ranking_)))


# In[86]:


col = X_train_balanced.columns[rfe.support_]
print(col)
col_rfe = col


# In[ ]:





# # Accessing the model with the StatsModel

# In[87]:


X_train_sm = sm.add_constant(X_train_balanced[col])
logm4 = sm.GLM(y_train_balanced, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[88]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[89]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# ## Creating a dataframe with the actual churn flag and the predicted probabilities

# In[90]:


y_train_pred_final = pd.DataFrame({'Churn':y_train_balanced.churn.values, 'Churn_Prob':y_train_pred})


# In[91]:


y_train_pred_final['CustID'] = y_train_balanced.index
y_train_pred_final.head()


# In[92]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_balanced[col].columns
vif['VIF'] = [variance_inflation_factor(X_train_balanced[col].values, i) for i in range(X_train_balanced[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[1237]:


col


# In[93]:


col = col.drop('loc_ic_t2m_mou_8')


# In[94]:


col


# In[95]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train_balanced[col])
logm3 = sm.GLM(y_train_balanced,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[96]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[97]:


y_train_pred[:10]


# In[98]:


y_train_pred_final['Churn_Prob'] = y_train_pred


# In[99]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[100]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# In[101]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
print(confusion)


# In[102]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_balanced[col].columns
vif['VIF'] = [variance_inflation_factor(X_train_balanced[col].values, i) for i in range(X_train_balanced[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### This dataset was run for many models. The good models have both 6 as well as 7 attributes. In most model, we needed to remove the attributes 2 times due to high vif for 2 attributes. The latest model requires only 1 attribute to be removed. So, commenting out the following code instead of removing it for future ref. As the next might give us a model with 6 attributes.

# In[103]:


#col = col.drop('loc_ic_mou_8')
#col


# In[104]:


# Re-runing the model using the selected variables
#X_train_sm = sm.add_constant(X_train_balanced[col])
#logm4 = sm.GLM(y_train_balanced,X_train_sm, family = sm.families.Binomial())
#res = logm4.fit()
#res.summary()


# In[105]:


#y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[106]:


#y_train_pred[:10]


# In[107]:


#y_train_pred_final['Churn_Prob'] = y_train_pred


# In[108]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
#y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
#y_train_pred_final.head()


# In[109]:


# Checking the overall accuracy.
#print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# In[110]:


# Confusion matrix 
#confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
#print(confusion)


# In[111]:


# Checking for the VIF values of the feature variables. 

#Create a dataframe that will contain the names of all the feature variables and their respective VIFs
#vif = pd.DataFrame()
#vif['Features'] = X_train_balanced[col].columns
#vif['VIF'] = [variance_inflation_factor(X_train_balanced[col].values, i) for i in range(X_train_balanced[col].shape[1])]
#vif['VIF'] = round(vif['VIF'], 2)
#vif = vif.sort_values(by = "VIF", ascending = False)
#vif


# #### All variables have a good value of VIF. So we need not drop any more variables

# # Finding optimal cutoff point
# 

# ## Optimal cutoff probability is that prob where we get balanced sensitivity and specificity and high value of sensitivity

# In[112]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[113]:


# Calculating accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[114]:


# Plotting accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# ### From the curve above, 0.55 is the optimum point to take it as a cutoff probability.

# In[115]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.55 else 0)

y_train_pred_final.head()


# In[116]:


# Checking the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.final_predicted)


# In[117]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )
confusion2


# # The sensitivity is around 84% on an average which is acceptable

# In[119]:


col


# # Making Predictions

# In[120]:


# Running the model using the selected variables


log_reg_2 = LogisticRegression(random_state=50)
log_reg_2.fit(X_train_balanced[col], y_train_balanced)


# In[121]:


y_log_reg_2_train_prediction = log_reg_2.predict(X_train_balanced[col])
log_reg_2_confusion_matrix = metrics.confusion_matrix(y_train_balanced, y_log_reg_2_train_prediction)


# In[122]:


print(log_reg_2_confusion_matrix)
print(metrics.classification_report(y_train_balanced, y_log_reg_2_train_prediction))


# ### Even with the same attributes which we got from the stats model and using that for logistics regression, the values like sensitivity and accuracy is good 

# In[123]:


X_test[col].head()


# In[124]:


# Predicted probabilities
y_pred = log_reg_2.predict_proba(X_test[col])
# Converting y_pred to a dataframe which is an array
y_pred_df = pd.DataFrame(y_pred)
# Converting to column dataframe
y_pred_1 = y_pred_df.iloc[:,[1]]
# Let's see the head
y_pred_1.head()


# In[125]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test).copy()
y_test_df.head()


# In[126]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 1 : 'Churn_Prob'})
# Rearranging the columns
#y_pred_final = y_pred_final.reindex_axis(['churn','Churn_Prob'], axis=1)
# Let's see the head of y_pred_final
y_pred_final.head()


# In[ ]:





# ## Creating new column 'predicted' with 1 if Churn_Prob>0.55 else 0 as taken from the results from the train ddata

# In[127]:



y_pred_final['predicted'] = y_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.55 else 0)
# Let's see the head
y_pred_final.head()


# # Model Evaluation

# In[ ]:





# In[128]:


from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# In[129]:


# Confusion matrix 
print(metrics.confusion_matrix( y_pred_final.churn, y_pred_final.predicted ))
print(metrics.classification_report(y_pred_final.churn, y_pred_final.predicted))


# In[130]:


# Checking the overall accuracy.
metrics.accuracy_score(y_pred_final.churn, y_pred_final.predicted)


# In[131]:


# Drawing the ROC curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


# In[132]:


draw_roc(y_pred_final.churn, y_pred_final.predicted)


# In[133]:


"{:2.2f}".format(metrics.roc_auc_score(y_pred_final.churn, y_pred_final.Churn_Prob))


# In[ ]:





# ## We see that we can good recall and accuracy here with the test data. So, our model is acceptable for the data we have

# In[ ]:





# # Inferences from the Logistic Regression with rfe and statsmodel

# ## Accuracy is around 80%. The aim of above model is to find out most number of the churners. So, the model was more focussed on accuracy as well as good sensitivity/recall. Our model achieved that goal.  We achieved a good sensitivity with good percentage for other important parameters.

# ## We created 2 models and the features accordingly are mentioned below:

# In[ ]:


### Model here


# # PCA on the data

# In[ ]:





# In[137]:


X_train_balanced.shape


# In[138]:


# Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)


# In[139]:


#Doing the PCA on the train data
pca.fit(X_train_balanced)


# In[ ]:





# # Plotting the principal components

# In[140]:


pca.components_


# In[141]:


# The variance explained


# In[142]:


pca.explained_variance_ratio_


# In[143]:


colnames = list(X_train_balanced.columns)
pca_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':colnames})
pca_df.head()


# In[144]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(pca_df.Feature):
    plt.annotate(txt, (pca_df.PC1[i],pca_df.PC2[i]))
plt.tight_layout()
plt.show()


# In[145]:


pca.explained_variance_ratio_


# In[146]:


# Making the screeplot - plotting the cumulative variance against the number of components

fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[ ]:





# ### We will chose 30 components as they explain 90% of the variance in the dataset

# In[ ]:





# In[147]:


# Using incremental PCA for efficiency - saves a lot of time on larger datasets
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=30)


# ## Basis transformation - getting the data onto our PCs

# In[148]:


df_train_pca = pca_final.fit_transform(X_train_balanced)
df_train_pca.shape


# In[149]:


pca_final.components_


# In[150]:


colnames = list(X_train_balanced.columns)
pca_df_2 = pd.DataFrame({'PC1':pca_final.components_[0],'PC2':pca_final.components_[1], 'Feature':colnames})
pca_df_2.head(20)


# In[ ]:





# ## Creating correlation matrix for the principal components - we expect little to no correlation

# In[151]:


#creating correlation matrix for the principal components
corrmat = np.corrcoef(df_train_pca.transpose())


# In[152]:


#plotting the correlation matrix
#%matplotlib inline
plt.figure(figsize = (20,10))
sns.heatmap(corrmat,annot = True)


# In[153]:


# 1s -> 0s in diagonals
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
# we see that correlations are indeed very close to 0


# ### There seems to be no correlation here

# In[154]:


#Applying selected components to the test data - 16 components
df_test_pca = pca_final.transform(X_test)
df_test_pca.shape


# ## Applying a logistic regression on our Principal Components

# In[155]:


#Training the model on the train data

learner_pca = LogisticRegression(C=1e9)
model_pca = learner_pca.fit(df_train_pca,y_train_balanced)


# # Making prediction on test data

# In[156]:


y_pred_pca = learner_pca.predict(df_test_pca)
df_y_pred_pca = pd.DataFrame(y_pred_pca)
print(confusion_matrix(y_test,y_pred_pca))
print("Accuracy with Logistic Regression using PCA for 30 features:",metrics.accuracy_score(y_test,y_pred_pca))


# In[157]:


#Making prediction on the test data
pred_probs_test = model_pca.predict_proba(df_test_pca)[:,1]
#print(metrics.roc_auc_score(y_test, pred_probs_test))


# In[158]:


pred_probs_test


# In[159]:


"{:2.2}".format(metrics.roc_auc_score(y_test, pred_probs_test))


# #### Trying out the automatic feature selection for PCA

# In[160]:


pca_again = PCA(0.85)


# In[161]:


df_train_pca2 = pca_again.fit_transform(X_train_balanced)
df_train_pca2.shape
# we see that PCA selected 14 components


# In[162]:


#training the regression model
learner_pca2 = LogisticRegression()
model_pca2 = learner_pca2.fit(df_train_pca2,y_train_balanced)


# In[163]:


df_test_pca2 = pca_again.transform(X_test)
df_test_pca2.shape


# In[164]:


#Making prediction on the test data
pred_probs_test2 = model_pca2.predict_proba(df_test_pca2)[:,1]
"{:2.2f}".format(metrics.roc_auc_score(y_test, pred_probs_test2))


# In[ ]:





# In[165]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
plt.scatter(df_train_pca[:,0], df_train_pca[:,1], c = y_train_balanced['churn'].map({0:'green',1:'red'}))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()


# In[ ]:





# # Results after Logistic Regression using PCA:

# ## We have a good result here. We have a accuracy of 81% with a good sensitivity score

# In[ ]:





# # RFE with Logistics Regression and stats model has been tried out.
# # PCA with logistics regression has been tried out

# #  Trying out a decision tree model 

# In[166]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[167]:


random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train, y_train)


# In[168]:


df_random_forest_preds = random_forest_classifier.predict(X_test)


# In[169]:


print(metrics.classification_report(y_test,df_random_forest_preds))
print(metrics.confusion_matrix(y_test, df_random_forest_preds))
print(metrics.accuracy_score(y_test, df_random_forest_preds))


# # The default parameters are not giving good result for sensitivity

# # Hyperparameters Tuning

# In[170]:


X_train.shape


# ### Tuning max_depth

# In[171]:


# GridSearchCV to find optimal n_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}

# instantiate the model
rf = RandomForestClassifier()


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="recall", return_train_score=True)
rf.fit(X_train, y_train)


# In[172]:


# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[173]:


# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training sensitivity")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test sensitivity")
plt.xlabel("max_depth")
plt.ylabel("Recall")
plt.legend()
plt.show()


# ### WE see above that after a point, the model tries to overfit.  

# In[ ]:





# ## Tuning n_estimators

# In[174]:


# Deliberately commented out the following code after running it once


# In[175]:


# GridSearchCV to find optimal n_estimators
#from sklearn.model_selection import KFold
#from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
#n_folds = 5

# parameters to build the model on
#parameters = {'n_estimators': range(100, 1500, 400)}


#rf = RandomForestClassifier(max_depth=4)


# fit tree on training data
#rf = GridSearchCV(rf, parameters, 
#                    cv=n_folds, 
#                   scoring="accuracy" , return_train_score=True)
#rf.fit(X_train, y_train)


# In[176]:


# scores of GridSearch CV
#scores = rf.cv_results_
#pd.DataFrame(scores).head()


# In[177]:


# plotting accuracies with n_estimators
#plt.figure()
#plt.plot(scores["param_n_estimators"], 
#         scores["mean_train_score"], 
#         label="training accuracy")
#plt.plot(scores["param_n_estimators"], 
#         scores["mean_test_score"], 
#         label="test accuracy")
#plt.xlabel("n_estimators")
#plt.ylabel("Accuracy")
#plt.legend()
#plt.show()


# 

# ## Tuning max_features

# In[178]:


# GridSearchCV to find optimal max_features
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_features': [4, 8, 14, 20, 24]}

# instantiate the model
rf = RandomForestClassifier(max_depth=4)


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="recall",return_train_score=True)
rf.fit(X_train, y_train)


# In[179]:


# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()


# In[180]:


# plotting accuracies with max_features
plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training sensitivity")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test sensitivity")
plt.xlabel("max_features")
plt.ylabel("Sensitivity")
plt.legend()
plt.show()


# # 15 can be a good number of features here

# In[181]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [6,8],
    'max_features': [10, 15]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1, return_train_score=True)


# In[182]:


grid_search.fit(X_train, y_train)


# In[183]:


print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# In[184]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [6],
    'max_features': [15]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1, return_train_score=True, scoring='recall')


# In[185]:


grid_search.fit(X_train, y_train)


# In[186]:


# scores of GridSearch CV
scores = grid_search.cv_results_
pd.DataFrame(scores).head()


# In[187]:


print('We can get sensitivity of',grid_search.best_score_,'using',grid_search.best_params_)


# In[188]:


# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=6,
                             max_features=15,
                             )


# In[189]:


rfc.fit(X_train,y_train)


# In[190]:


predictions = rfc.predict(X_test)


# In[191]:


# evaluation metrics
from sklearn.metrics import classification_report,confusion_matrix


# In[192]:


print(classification_report(y_test,predictions))


# In[193]:


print(confusion_matrix(y_test,predictions))


# # Trying out with the features obtained from the rfe and statsmodel

# In[194]:


col


# In[195]:


# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[196]:


# fit
rfc.fit(X_train[col],y_train)


# In[197]:


# Making predictions
predictions = rfc.predict(X_test[col])


# In[198]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[199]:


print(confusion_matrix(y_test,predictions))


# In[ ]:





# # Trying out with the features obtained from the rfe

# In[200]:


col_rfe


# In[201]:


# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[202]:


# fit
rfc.fit(X_train[col_rfe],y_train)


# In[203]:


# Making predictions
predictions = rfc.predict(X_test[col_rfe])


# In[204]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[205]:


print(confusion_matrix(y_test,predictions))


# ### Clearly decision is not something we should consider here if we are looking for good recall

# ### It is observed that the decision can provide us good measure of accuracy but it has poor sensitivity percentage

# In[ ]:





# # Final results

# ### We aimed to have good percentage of sensitivity/recall as per the usecase.

# ## Models tried:
# * Logistics regression with rfe and statsmodel
# * Logistics regression with PCA
# * Random forests
# * Random forests with RFE

# ### Random is not a good choice here. Although random forest have good accuracy but it lacks in sensitivity percentage which is not ideal for our scenario

# ## The model logistic regression with rfe and statsModel and Logistics Regression with PCA 

# ### The most important parameters which drives the churn are:
# ##### fields here
# 
# ##### The features above are the combination of many models. 4 of which is listed in the cell after the logistic regression with rfe.
# ##### The exact features in each of the model can be referenced from the cell after the logistic regression with rfe above.

# ## Inferences:

# ## The following actions drives the whole telecom data with the possible reasons for churning??
# 
# 

# # The factors from the other operators which can influence the churn:
# * Low local call rates
# * Low std call rates
# * Low charges for facebook and internet packs
# * Low speed for internet

# # Strategies to reduce churn:
# 
# * Give offers and discounts to users on the usage basis
# * Provide special packs and discounts for users with high call numbers within the same network
# * Provide special packs and discounts for users with high call numbers outside the network
# * Provide good data packs for the users of high usage
# * Make sure that the customer service is good

# In[ ]:




