#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# In[ ]:


#skidanje dataset-a
path = kagglehub.dataset_download("khwaishsaxena/lung-cancer-dataset")

print("Path to dataset files:", path)


# In[ ]:


#učitavanje datoteke
file_path = 'C:\\Users\\Zika\\Desktop\\skladistenje-softver\\Lung Cancer.csv'
df = pd.read_csv(file_path)


# In[ ]:


#prikaz po
df.head()


# In[ ]:


# prikaz broja redova
df.shape[0]


# In[ ]:


# prikaz broja kolona
df.shape[1]


# In[ ]:


# detaljniji prikaz informacija o tipu i broja null podataka
df.info()


# In[ ]:


# koriscenje ydata profiliga za analizu i odredjenu vizualizaciju podtaka podataka 
profile = ProfileReport(df)
profile.to_notebook_iframe()


# In[16]:


df.columns


# In[17]:


df.isnull().sum()


# In[18]:


df.describe()


# In[19]:


df['survived'].value_counts()


# In[20]:


df['age'].nunique()


# In[21]:


df['cholesterol_level'].hist()


# In[22]:


sns.boxplot(data=df, x='bmi')


# In[23]:


df_numeric = df.select_dtypes(include='number').drop(["id"], axis=1) 

df_numeric.corr()


# In[24]:


df['bmi'].value_counts(normalize=True)


# In[25]:


df['datum_dijagnoze'] = pd.to_datetime(df['diagnosis_date'])
df['kraj_tretmana'] = pd.to_datetime(df['end_treatment_date'])

df['treatment_duration_days'] = (df['kraj_tretmana'] - df['datum_dijagnoze']).dt.days
df.drop(['diagnosis_date', 'end_treatment_date', 'kraj_tretmana', 'datum_dijagnoze'], axis=1, inplace=True)


# In[26]:


columns_to_encode = ['gender', 'country', 'family_history', 'smoking_status', 'treatment_type']

df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)


# In[27]:


df['cancer_stage'].value_counts()


# In[28]:


stage_order = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

df['stage'] = pd.Categorical(df['cancer_stage'], categories=stage_order, ordered=True).codes
df.drop('cancer_stage', axis=1, inplace=True)


# In[29]:


df.head()


# In[30]:


df = df.astype({col: 'int' for col in df.select_dtypes('bool').columns})


# In[31]:


df.head()


# In[34]:


X = df.drop(['id','survived'],axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)


# In[35]:


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


# In[36]:


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:


conf_matrix_xg = confusion_matrix(y_test, y_pred)
print("🧾 Confusion Matrix:")
print(conf_matrix_xg)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_xg, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[52]:


rf_model = RandomForestClassifier(
    n_estimators=10,        
    max_depth=50,            
    class_weight='balanced',
    random_state=52,         
    n_jobs=-1                
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)


# In[ ]:


conf_matrix_rf = confusion_matrix(y_test, rf_predictions)
print("🧾 Confusion Matrix:")
print(conf_matrix_rf)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[54]:


importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]


# In[56]:


plt.figure(figsize=(8,6))
sns.barplot(x=importances[indices[:10]], y=X_train.columns[indices[:10]])
plt.title(f'Top 10 Feature Importances in XGBoost')
plt.show()

