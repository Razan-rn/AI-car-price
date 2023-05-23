#!/usr/bin/env python
# coding: utf-8

# In[186]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('CarPrice_المشروع الثاني.csv')
df.head()


# In[187]:


df.info()


# In[188]:


df.nunique()


# In[189]:


df.CarName.unique()


# In[190]:


df['CarName'].value_counts()#بيانات العناصر في هذا العمود عدد تكررها


# In[191]:


df['CarName'] = df['CarName'].str.split(' ',expand = True)[0]


# In[192]:


df['CarName'].unique()


# In[193]:


df['CarName'] = df['CarName'].replace({'maxda':'mazda','nassan':'Nissan','porcshce':'porsche','toyouta':'toyota','vokswagen':'volkswagen','vw':'volkswagen'})


# In[194]:


df['CarName'].unique()


# In[195]:


plt.figure(figsize=(15,15))
ax = sns.countplot(x=df['CarName']);
ax.bar_label(ax.containers[0]);
plt.xticks(rotation=90);


# In[196]:


sns.set_style('whitegrid')
plt.figure(figsize=(15,10))
sns.distplot(df.price)
plt.show()


# In[197]:


ax = sns.countplot(x=df['fueltype']);
ax.bar_label(ax.containers[0]);


# In[198]:


# import seaborn as sns
# sns.pairplot(df,markers=None,hue='price')
# plt.show()


# In[199]:


new_df = df[['fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem','wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']]
new_df.head()


# In[200]:


new_df = pd.get_dummies(columns=['fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem'],data=new_df)
new_df.head()


# In[201]:


scaler = StandardScaler()
num_cols = ['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg']
new_df[num_cols] = scaler.fit_transform(new_df[num_cols])


# In[202]:


x = new_df.drop(columns=['price'])
y = new_df['price']
x.shape


# In[203]:


y.shape


# In[204]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[205]:


treining_score= []
testing_score= []


# In[206]:


from sklearn. metrics import r2_score

def model_prediction(model):
    model.fit(x_train,y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred=model.predict(x_test)
    a= r2_score (y_train, x_train_pred)*100 
    b= r2_score (y_test,x_test_pred)*100
    treining_score.append(a) 
    testing_score.append(b)
    print(f"r2_Score of (model) model on Training Data is:",a) 
    print(f"r2_Score of (model) model on Testing Data is: ",b)


# In[207]:


from sklearn.linear_model import LinearRegression
model_prediction(LinearRegression())


# In[208]:


from sklearn.tree import DecisionTreeRegressor
model_prediction(DecisionTreeRegressor())


# In[209]:


from sklearn.ensemble import RandomForestRegressor
model_prediction(RandomForestRegressor())


# In[210]:


#!pip install catboost


# In[212]:


from catboost import CatBoostRegressor
model_prediction(CatBoostRegressor(verbose=False))


# In[213]:


models = ['Linear Regression','Decision Tree','Random Forest','Cat Boos']


# In[214]:


df1 = pd.DataFrame({'Algorrithms':models,
                   'Training Score':treining_score,
                   'Testing Score':testing_score})
df1


# In[215]:


df1.plot(x='Algorrithms',y=['Training Score','Testing Score'],figsize=(16,6),kind='bar', title='Performance Visualهzation of Different Models',colormap='Set1')
plt.show()


# In[ ]:




