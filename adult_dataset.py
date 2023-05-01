#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

import qgrid

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

train = pd.read_hdf('../input/train.adult.h5')

# ## Zadanie 1.5.3

# In[39]:

print(train.columns)
train.info()

plt.figure(figsize=(15, 5))
feats = train.select_dtypes(include=[np.object]).columns[:-1]
feats

for feat in feats:
    plt.figure(figsize=(15, 5))
    plt.title(feat)
    #train[feat].value_counts().plot(kind='bar')
    sns.countplot(x=feat, data=train);
    plt.xticks(rotation=90);
    plt.show();

# ```python
# print(train.columns)
# 
# 
# plt.figure(figsize=(15, 5))
# sns.countplot(x='Workclass', data=train);
# plt.xticks(rotation=90);
# plt.show()
# 
# plt.figure(figsize=(15, 5))
# sns.countplot(x='Martial Status', data=train);
# plt.xticks(rotation=90);
# plt.show()
# 
# plt.figure(figsize=(15, 5))
# sns.countplot(x='Relationship', data=train);
# plt.xticks(rotation=90);
# plt.show()
# 
# ```
#  
# </p>
# </details>
# </p>
# </details> 

# In[40]:

plt.figure(figsize=(15, 5))
sns.barplot(x="Education", y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# In[41]:

plt.figure(figsize=(15, 5))
sns.barplot(x="Race", y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy kraj pochodzenia oraz płeć.

# In[42]:


plt.figure(figsize=(15, 5))
sns.barplot(x="Country", y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy stan cywilny oraz płeć.

# In[43]:

plt.figure(figsize=(15, 5))
sns.barplot(x='Martial Status', y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy zawód oraz płeć.

# In[44]:


plt.figure(figsize=(15, 5))
sns.barplot(x='Occupation', y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# In[45]:


plt.figure(figsize=(15, 5))
sns.barplot(x='Occupation', y="Target_cat", hue='Race', data=train)
plt.xticks(rotation=90);


# Zobacz jeszcze jedną wskazówkę co do wizualizacji. Możesz rozbić to na osobne wykresy.

# In[46]:


plt.figure(figsize=(20, 5))
g = sns.catplot(x="Occupation", y="Target_cat", col="Sex", data=train, kind="bar")

for ax in g.axes.flatten():
    plt.sca(ax)
    plt.xticks(rotation=90)


# Jeśli jest więcej niż 5, to można powiedzieć, ile ma być maksymalnie w jednym wierszu (używając `col_wrap`).

# In[47]:


plt.figure(figsize=(20, 5))
g = sns.catplot(x="Occupation", y="Target_cat", hue="Sex", col='Race', col_wrap=3, data=train, kind="bar")

for ax in g.axes.flatten():
    plt.sca(ax)
    plt.xticks(rotation=90)


# ## Zadanie 1.5.4
# Dodaj kolejne cechy (*features*) bazując na tych, które już są (np. na podstawie dwóch cech `Relationship` i `Race`, można wykombinować kilka różnych cech, np. `White` + `Husband` lub `Black` + `Husband`).

# Daj znać na Slacku, czy udało Ci się sprawdzić tę cechę :) 

# In[48]:

train['Black_Husband'] = train['Race'].map(lambda y: int(y == 'Black')) * train['Relationship'].map(lambda z: int(z == 'Husband'))

plt.figure(figsize=(10,4))
sns.barplot(x='Black_Husband', y="Target", data=train)
plt.xticks(rotation=90)


# In[1]:

train["relationship_race"] = train.apply(lambda x: "{}-{}".format(x["Relationship"], x["Race"]), axis=1)
train["relationship_race_cat"] = train["relationship_race"].factorize()[0]

# ## Zadanie 1.5.5
# Zastosuj bardziej złożony model, np. [DecisionTreeClassifier](https://bit.ly/39qD4Vk). Dlaczego akurat ten? Bo jest relatywnie prosty, ale znacznie skuteczniejszy niż `Dummy` model. Więcej o drzewach decyzyjnych będzie w następnych modułach. Dlatego na razie możesz to potraktować jako czarne pudło.

# In[4]:

from sklearn.tree import DecisionTreeClassifier
def train_and_predict_model(X_train, X_test, y_train, y_test, model, success_metric=accurancy_score):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return success_metric(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random.state=2018)
lr = train_and_predict_model(X_train, X_test, y_train, y_test, LogisticRegression())
print('LogisticRegression: ', lr)
print('')
dtc = train_and_predict_model(X_train, X_test, y_train, y_test, DecisionTreeClassifier())
print('DecisionTreeClassifier: ', dtc)
train.columns
