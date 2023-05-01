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


# ## Wczytujemy dane
# Dane są przechowywane w formacie [HDF5](https://bit.ly/3w6jbwk). To jest binarny format, który jest dość wygodny (zwykle trzymamy dane w tym formacie zamiast .csv). Między innymi umożliwia to zapisanie więcej niż jeden zbiór danych do jednego pliku.

# In[2]:


train = pd.read_hdf('../input/train.adult.h5')

# ## Zadanie 1.5.3
# 
# Spróbuj zbadać kolejne cechy, to może być przydatne, żeby zacząć tworzyć lepsze cechy.

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


# 
# 
# <details>
#     <summary style="background: #e6eaeb; padding: 4px 0; text-align: center; font-size: 20px; font-weight: 900;"> 👉 Kliknij tutaj (1 klik), aby zobaczyć odpowiedź 👈 </summary>
# <p>
# 
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

# Zobaczmy, jak wygląda rozkład danych. To pomoże Ci zobaczyć pewne zależności i zacząć tworzyć kolejne cechy (oparte na kombinacji istniejących cech).
# 
# Zaczniemy od zbadania: płci oraz edukacji. Przypominam, że na osi y, jest prawdopodobieństwo, że ta osoba zarobi więcej niż 50k rocznie (1.0 oznacza 100%).

# In[40]:


plt.figure(figsize=(15, 5))
sns.barplot(x="Education", y="Target_cat", hue='Sex', data=train)
plt.xticks(rotation=90);


# Zbadajmy teraz rasę i płeć.

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


# Mam nadzieję, że już widać, które grupy wyróżniają się (a to brzmi jak cecha). Tu przy okazji zbadaliśmy temat niesprawiedliwości tego świata. Co jest ważne, model nic nie wie na temat dyskryminacji, jedynie uczy się z tego co jest "myśląc", że to jest normalne. Stąd właśnie pojawia się bias, zobacz ten [filmik](https://www.youtube.com/watch?v=59bMh59JQDo). Musisz na to uważać! Model staje się tym, czym go karmisz :). Podobnie jak nasz mózg (to co tam wrzucamy, wpływa na to, kim się stajemy później).
# 
# 
# Zbadaj teraz jeszcze inne kombinacje np. zamiast płci sprawdź rasę.

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


# Zwróć uwagę na czarne pionowe kreski (w słupkach). Traktuj to jak rozrzut danych, jeśli ta kreska jest zbyt zmienna (np. zobacz ostatni wykres), to ciężko cokolwiek wnioskować, bo jest zbyt duża zmienność.
# 
# Już sporo wiesz, żeby dalej poruszać się samodzielnie. Spróbuj zrobić następujące rzeczy.

# ## Zadanie 1.5.4
# Dodaj kolejne cechy (*features*) bazując na tych, które już są (np. na podstawie dwóch cech `Relationship` i `Race`, można wykombinować kilka różnych cech, np. `White` + `Husband` lub `Black` + `Husband`).
# 
# Moje oczekiwania są takie, że spróbujesz stworzyć kilka czy kilkanaście nowych cech. Od razu powiem, przygotuj się, że często nowa cecha może być mało wartościowa. Natomiast wartością będzie, jeśli nauczysz się szybko iterować hipotezy (czyli odkrywać nowe cechy, które wnoszą wartość poprzez szybkie eksperymenty).
# 
# Ciekawostka: być może warto dodać zmienną waga? Zobacz ten artykuł: [People who are overweight get paid less, according to a new LinkedIn study](https://bit.ly/39njuch)

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
